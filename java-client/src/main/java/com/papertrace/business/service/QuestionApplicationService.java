package com.papertrace.business.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.QuestionResponse;
import com.papertrace.business.api.QuestionResponse.*;
import com.papertrace.internal.PaperTracePythonClient;
import java.io.IOException;
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
                            new Validation(quoteValidation, semantic, basis)));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new QuestionServiceException("问答内部服务请求已中断", error);
        } catch (IOException | RuntimeException error) {
            throw new QuestionServiceException("问答内部服务不可用", error);
        }
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
        /** 创建不携带原始问题或模型正文的业务异常。 */
        public QuestionServiceException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
