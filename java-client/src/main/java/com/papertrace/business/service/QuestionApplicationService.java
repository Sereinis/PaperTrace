package com.papertrace.business.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.QuestionResponse;
import com.papertrace.business.api.QuestionResponse.*;
import com.papertrace.internal.PaperTracePythonClient;
import java.io.IOException;
import java.net.http.HttpTimeoutException;
import com.papertrace.internal.PaperTracePythonClient.PythonResponseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.springframework.stereotype.Service;

/** 编排 Java 业务请求与 Python 证据问答服务之间的调用。 */
@Service
public class QuestionApplicationService {
    private final PaperTracePythonClient pythonClient;
    private final ObjectMapper objectMapper;

    /** 注入内部客户端和 JSON 解析器。 */
    public QuestionApplicationService(PaperTracePythonClient pythonClient, ObjectMapper objectMapper) {
        this.pythonClient = pythonClient;
        this.objectMapper = objectMapper;
    }

    /** 调用 Python 服务并解析为业务层外壳，拒绝畸形成功响应。 */
    public QuestionResponse ask(String query, String documentId) {
        try {
            JsonNode envelope = objectMapper.readTree(pythonClient.ask(query, documentId));
            if (envelope == null || !envelope.path("ok").isBoolean() || !envelope.path("ok").booleanValue()
                    || !"papertrace-internal-question-v1".equals(text(envelope, "service_version"))) {
                throw new IllegalStateException("内部协议版本或成功状态无效");
            }
            JsonNode result = envelope.path("result");
            if (!query.equals(text(result, "query")) || !documentId.equals(text(result, "document_id"))) {
                throw new IllegalStateException("内部响应请求身份不一致");
            }
            String status = text(result, "status");
            if (!Set.of("answered", "partial", "insufficient").contains(status)) {
                throw new IllegalStateException("内部回答状态无效");
            }
            List<Claim> claims = new ArrayList<>();
            for (JsonNode claim : array(result, "claims")) {
                List<Citation> citations = new ArrayList<>();
                for (JsonNode citation : array(claim, "citations")) {
                    String sourceDocument = text(citation, "document_id");
                    int start = number(citation, "quote_start");
                    int end = number(citation, "quote_end");
                    int page = number(citation, "page_number");
                    if (!documentId.equals(sourceDocument) || start < 0 || end <= start || page < 1) {
                        throw new IllegalStateException("内部引用身份或位置无效");
                    }
                    citations.add(new Citation(text(citation, "evidence_id"), text(citation, "segment_id"),
                            text(citation, "chunk_id"), sourceDocument, page, text(citation, "quote"), start, end));
                }
                if (citations.isEmpty()) {
                    throw new IllegalStateException("内部结论缺少引用");
                }
                claims.add(new Claim(text(claim, "text"), List.copyOf(citations)));
            }
            List<String> missing = new ArrayList<>();
            for (JsonNode item : array(result, "missing_info")) {
                if (!item.isTextual() || item.textValue().isBlank()) {
                    throw new IllegalStateException("内部缺失信息无效");
                }
                missing.add(item.textValue());
            }
            if ((status.equals("answered") && (claims.isEmpty() || !missing.isEmpty()))
                    || (status.equals("partial") && (claims.isEmpty() || missing.isEmpty()))
                    || (status.equals("insufficient") && (!claims.isEmpty() || missing.isEmpty()))) {
                throw new IllegalStateException("内部回答状态与内容不一致");
            }
            JsonNode validation = result.path("validation");
            String quoteValidation = text(validation, "citation_identity_and_quote");
            String semantic = text(validation, "semantic_support");
            String basis = text(validation, "status_basis");
            if (!semantic.equals("not_verified")
                    || !Set.of("passed", "not_applicable").contains(quoteValidation)
                    || !Set.of("model_self_assessment", "no_retrieved_evidence").contains(basis)) {
                throw new IllegalStateException("内部校验状态无效");
            }
            return new QuestionResponse(text(envelope, "service_version"),
                    new Answer(query, documentId, status, List.copyOf(claims), List.copyOf(missing),
                            new Validation(quoteValidation, semantic, basis), usage(result.path("generation").path("usage"))));
        } catch (PythonResponseException error) {
            throw classify(error);
        } catch (HttpTimeoutException error) {
            throw new QuestionServiceException("internal_timeout", 504, null, error);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new QuestionServiceException("问答内部服务请求已中断", error);
        } catch (IOException | RuntimeException error) {
            throw new QuestionServiceException("问答内部服务不可用", error);
        }
    }

    /** 只接受固定版本与HTTP状态匹配的错误码，不使用上游message。 */
    private QuestionServiceException classify(PythonResponseException error) {
        try {
            JsonNode envelope = objectMapper.readTree(error.body());
            JsonNode detail = envelope.path("error");
            if (!envelope.path("ok").isBoolean() || envelope.path("ok").booleanValue()
                    || !text(detail, "service_version").equals("papertrace-internal-question-v1")) {
                throw new IllegalStateException();
            }
            String code = text(detail, "code");
            int expected = switch (code) {
                case "document_not_found" -> 404;
                case "invalid_input", "invalid_json", "invalid_content_length" -> 400;
                case "context_too_large" -> 422;
                case "service_busy" -> 503;
                case "retrieval_failed", "generation_failed", "citation_invalid",
                        "answer_contract_invalid", "invalid_generation_result" -> 502;
                default -> throw new IllegalStateException();
            };
            if (error.status() != expected) { throw new IllegalStateException(); }
            return new QuestionServiceException(code, expected, usage(detail.path("usage")), error);
        } catch (IOException | RuntimeException invalid) {
            return new QuestionServiceException("问答内部服务不可用", error);
        }
    }

    /** 校验生成用量的非负整数及求和关系，未知保持为空。 */
    private static Usage usage(JsonNode node) {
        if (node.isMissingNode() || node.isNull()) { return null; }
        String[] names = {"prompt_tokens", "completion_tokens", "total_tokens"};
        long[] values = new long[3];
        for (int i = 0; i < names.length; i++) {
            JsonNode value = node.path(names[i]);
            if (!value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() < 0) {
                throw new IllegalStateException("内部用量结构无效");
            }
            values[i] = value.longValue();
        }
        if (values[0] > Long.MAX_VALUE - values[1] || values[0] + values[1] != values[2]) {
            throw new IllegalStateException("内部用量求和无效");
        }
        return new Usage(values[0], values[1], values[2]);
    }

    /** 提取必需的非空字符串，禁止 JSON 类型强制转换。 */
    private static String text(JsonNode object, String field) {
        JsonNode value = object.path(field);
        if (!value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalStateException("内部响应缺少字符串字段");
        }
        return value.textValue();
    }

    /** 提取必需的数组字段。 */
    private static JsonNode array(JsonNode object, String field) {
        JsonNode value = object.path(field);
        if (!value.isArray()) {
            throw new IllegalStateException("内部响应缺少数组字段");
        }
        return value;
    }

    /** 提取必需的整数，避免小数被静默截断。 */
    private static int number(JsonNode object, String field) {
        JsonNode value = object.path(field);
        if (!value.isIntegralNumber() || !value.canConvertToInt()) {
            throw new IllegalStateException("内部响应整数无效");
        }
        return value.intValue();
    }

    /** 将内部调用失败转换为业务层可处理异常。 */
    public static final class QuestionServiceException extends RuntimeException {
        private final String code;
        private final int httpStatus;
        private final Usage usage;
        /** 创建不携带原始问题或模型正文的业务异常。 */
        public QuestionServiceException(String message, Throwable cause) {
            super(message, cause);
            this.code = "internal_service_unavailable";
            this.httpStatus = 502;
            this.usage = null;
        }

        /** 保存经白名单检查的分类、HTTP状态与合法用量。 */
        public QuestionServiceException(String code, int httpStatus, Usage usage, Throwable cause) {
            super("问答请求未完成；不会自动重试", cause);
            this.code = code;
            this.httpStatus = httpStatus;
            this.usage = usage;
        }

        /** 获取安全分类。 */
        public String code() { return code; }

        /** 获取业务HTTP状态。 */
        public int httpStatus() { return httpStatus; }

        /** 获取已验证用量或未知值。 */
        public Usage usage() { return usage; }
    }
}
