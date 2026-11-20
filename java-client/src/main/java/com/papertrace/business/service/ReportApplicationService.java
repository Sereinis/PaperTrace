package com.papertrace.business.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.*;
import com.papertrace.business.api.ReportResponse.*;
import com.papertrace.internal.PaperTracePythonClient;
import com.papertrace.business.service.QuestionApplicationService.QuestionServiceException;
import java.util.*;
import org.springframework.stereotype.Service;

/** 代理异步报告任务，校验上游身份、字段顺序、引用与用量后投影。 */
@Service
public class ReportApplicationService {
    private final PaperTracePythonClient client;
    private final ObjectMapper mapper;
    private static final List<String> FIELDS = List.of("research_question", "datasets", "model", "baselines",
            "training_settings", "evaluation_metrics", "results");
    private static final List<String> LABELS = List.of("研究问题", "数据集", "模型", "对照方法", "训练设置", "评价指标", "实验结果");

    /** 注入受限本机客户端，不保存密钥。 */
    public ReportApplicationService(PaperTracePythonClient client, ObjectMapper mapper) {
        this.client = client; this.mapper = mapper;
    }

    /** 提交一次幂等任务，不重试失败或不确定结果。 */
    public ReportResponse submit(ReportRequest request) {
        try {
            if (new HashSet<>(request.resultTarget().requiredConditions()).size() != request.resultTarget().requiredConditions().size())
                throw new QuestionServiceException("invalid_input", 400, null, null);
            var payload = mapper.createObjectNode();
            payload.put("request_id", request.requestId()).put("document_id", request.documentId())
                    .put("experiment_scope", request.experimentScope());
            var target = payload.putObject("result_target");
            target.put("model", request.resultTarget().model()).put("task", request.resultTarget().task())
                    .put("dataset", request.resultTarget().dataset()).put("split", request.resultTarget().split())
                    .put("model_type", request.resultTarget().modelType());
            target.set("required_conditions", mapper.valueToTree(request.resultTarget().requiredConditions()));
            var envelope = call(mapper.writeValueAsString(payload));
            var node = envelope.path("task");
            if (!payload.equals(node.path("request"))) throw new IllegalStateException();
            return project(node);
        } catch (QuestionServiceException error) { throw error; }
        catch (Exception error) { throw unavailable(error); }
    }

    /** 只读后台任务，不创建新模型调用。 */
    public List<ReportResponse> list() {
        try {
            var tasks = call(null).path("tasks");
            if (!tasks.isArray() || tasks.size() > 50) throw new IllegalStateException();
            var result = new ArrayList<ReportResponse>();
            var ids = new HashSet<String>();
            for (var task : tasks) {
                var item = project(task);
                if (!ids.add(item.requestId())) throw new IllegalStateException();
                result.add(item);
            }
            return List.copyOf(result);
        } catch (QuestionServiceException error) { throw error; }
        catch (Exception error) { throw unavailable(error); }
    }

    /** 验证任务阶段与报告身份，禁止把未完成内容发布为结果。 */
    private ReportResponse project(JsonNode node) {
        var request = node.path("request");
        String id = text(request, "request_id"), document = text(request, "document_id"), scope = text(request, "experiment_scope");
        if (!id.matches("[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
                || document.length() > 256 || scope.length() > 1000) throw new IllegalStateException();
        String status = text(node, "status"), field = nullable(node, "current_field"), code = nullable(node, "error_code");
        int completed = integer(node, "completed_fields", 7);
        if (!Set.of("queued", "processing", "ready", "failed").contains(status)
                || (field != null && !FIELDS.contains(field))
                || (code != null && !Set.of("interrupted", "field_failed", "report_failed").contains(code)))
            throw new IllegalStateException();
        if ((status.equals("failed") && code == null) || (!status.equals("failed") && code != null)
                || (status.equals("ready") && (completed != 7 || field != null))
                || (status.equals("failed") && field != null)
                || (status.equals("queued") && (completed != 0 || field != null))) throw new IllegalStateException();
        Generation generation = generation(node.path("generation"));
        Result result = null;
        if (status.equals("ready")) {
            var raw = node.path("result");
            if (!document.equals(text(raw, "document_id")) || !scope.equals(text(raw, "experiment_scope"))
                    || !raw.path("sections").isArray() || raw.path("sections").size() != 7
                    || generation == null || !node.path("generation").equals(raw.path("generation"))) throw new IllegalStateException();
            var sections = new ArrayList<Section>();
            for (int i = 0; i < 7; i++) {
                var section = raw.path("sections").get(i);
                if (!FIELDS.get(i).equals(text(section, "field"))) throw new IllegalStateException();
                sections.add(new Section(FIELDS.get(i), LABELS.get(i),
                        QuestionApplicationService.parseAnswer(section.path("qa"), document, true)));
            }
            var validation = raw.path("validation");
            for (var key : List.of("semantic_support", "experiment_scope_match", "cross_field_consistency", "exhaustiveness")) {
                if (!"not_verified".equals(text(validation, key))) throw new IllegalStateException();
            }
            result = new Result(List.copyOf(sections), "not_verified", "not_verified", "not_verified", "not_verified");
        } else if (!node.path("result").isNull()) throw new IllegalStateException();
        return new ReportResponse(id, document, scope, status, completed, field, code, generation, result);
    }

    /** 校验调用计数、已知小计和完整用量之间的关系。 */
    private static Generation generation(JsonNode node) {
        if (node.isNull()) return null;
        int calls = integer(node, "generation_calls", 14), unknown = integer(node, "unknown_usage_calls", calls);
        var subtotal = QuestionApplicationService.usage(node.path("known_usage_subtotal"));
        var total = QuestionApplicationService.usage(node.path("total_usage"));
        if (subtotal == null || ((unknown > 0 || calls == 0) && total != null)
                || (calls > 0 && unknown == 0 && !subtotal.equals(total))) throw new IllegalStateException();
        return new Generation(calls, unknown, subtotal, total);
    }

    /** 严格读取非空文本字段。 */
    private static String text(JsonNode node, String key) {
        if (!node.path(key).isTextual() || node.path(key).textValue().isBlank()) throw new IllegalStateException();
        return node.path(key).textValue();
    }

    /** 可空文本必须显式为null而不是字段缺失。 */
    private static String nullable(JsonNode node, String key) {
        return node.path(key).isNull() ? null : text(node, key);
    }

    /** 校验有界非负整数，拒绝静默类型转换。 */
    private static int integer(JsonNode node, String key, int max) {
        var value = node.path(key);
        if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < 0 || value.intValue() > max)
            throw new IllegalStateException();
        return value.intValue();
    }

    /** 上游错误按固定码映射，不透传message或原始正文。 */
    private JsonNode call(String payload) {
        try {
            var envelope = mapper.readTree(client.reports(payload));
            if (envelope == null || !envelope.path("ok").isBoolean() || !envelope.path("ok").booleanValue()
                    || !"papertrace-internal-question-v1".equals(text(envelope, "service_version"))) throw new IllegalStateException();
            return envelope;
        } catch (PaperTracePythonClient.PythonResponseException error) {
            try {
                var envelope = mapper.readTree(error.body());
                var detail = envelope.path("error");
                String code = text(detail, "code");
                int status = switch (code) {
                    case "invalid_input", "invalid_json" -> 400;
                    case "document_not_found" -> 404;
                    case "service_busy", "reports_disabled" -> 503;
                    case "request_conflict", "report_limit" -> 409;
                    default -> throw new IllegalStateException();
                };
                if (!envelope.path("ok").isBoolean() || envelope.path("ok").booleanValue()
                        || !"papertrace-internal-question-v1".equals(text(detail, "service_version"))
                        || status != error.status()) throw new IllegalStateException();
                throw new QuestionServiceException(code, status, null, error);
            } catch (QuestionServiceException classified) { throw classified; }
            catch (Exception invalid) { throw unavailable(invalid); }
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt(); throw unavailable(error);
        } catch (Exception error) { throw unavailable(error); }
    }

    /** 依赖失败仅返回固定安全分类。 */
    private static QuestionServiceException unavailable(Exception cause) {
        return new QuestionServiceException("报告内部服务不可用", cause);
    }
}
