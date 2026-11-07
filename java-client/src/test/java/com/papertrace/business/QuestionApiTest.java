package com.papertrace.business;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** 使用真实本机 HTTP 与模拟 Python 响应验证完整 Java 业务入口。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class QuestionApiTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final AtomicInteger CALLS = new AtomicInteger();
    private static final HttpServer UPSTREAM = upstream();
    private static volatile int upstreamStatus;
    private static volatile String upstreamBody;
    private static volatile String received;
    @Autowired private TestRestTemplate rest;

    /** 启动本机模拟上游，记录请求并返回可控制的响应。 */
    private static HttpServer upstream() {
        try {
            HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/v1/question", exchange -> {
                CALLS.incrementAndGet();
                received = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                byte[] bytes = upstreamBody.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
                exchange.sendResponseHeaders(upstreamStatus, bytes.length);
                try (var output = exchange.getResponseBody()) {
                    output.write(bytes);
                }
            });
            server.start();
            return server;
        } catch (IOException error) {
            throw new IllegalStateException("无法启动测试上游", error);
        }
    }

    /** 将本机随机端口注入应用配置。 */
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("papertrace.internal-service-url", () -> "http://127.0.0.1:" + UPSTREAM.getAddress().getPort());
    }

    /** 恢复成功响应和调用计数。 */
    @BeforeEach
    void reset() {
        CALLS.set(0);
        upstreamStatus = 200;
        upstreamBody = """
                {"ok":true,"service_version":"papertrace-internal-question-v1","result":{
                "query":"attention","document_id":"a","status":"answered",
                "claims":[{"text":"模型使用注意力机制。","citations":[{
                "evidence_id":"E1","segment_id":"E1.S1","chunk_id":"a1","document_id":"a",
                "page_number":1,"quote":"The model uses attention.","quote_start":0,"quote_end":25}]}],
                "missing_info":[],"evidence":[{"text":"not exposed"}],"generation":{"request_id":"private"},
                "validation":{"citation_identity_and_quote":"passed","semantic_support":"not_verified",
                "status_basis":"model_self_assessment"}}}
                """;
    }

    /** 关闭本机模拟上游。 */
    @AfterAll
    static void close() {
        UPSTREAM.stop(0);
    }

    /** 调用业务接口并返回 JSON 响应。 */
    private ResponseEntity<JsonNode> ask(String body) {
        return rest.postForEntity("/api/questions", new HttpEntity<>(body, headers()), JsonNode.class);
    }

    /** 建立 JSON 请求头。 */
    private HttpHeaders headers() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return headers;
    }

    /** 验证 HTTP 链路、请求字段映射和强类型结果投影。 */
    @Test
    void returnsTypedProjection() throws Exception {
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().at("/result/claims/0/citations/0/chunkId").asText()).isEqualTo("a1");
        assertThat(response.getBody().at("/result/validation/semanticSupport").asText()).isEqualTo("not_verified");
        assertThat(response.getBody().path("result").has("evidence")).isFalse();
        assertThat(response.getBody().path("result").has("generation")).isFalse();
        assertThat(JSON.readTree(received).get("document_id").asText()).isEqualTo("a");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 空白输入在调用上游前拒绝。 */
    @Test
    void rejectsBlankInput() {
        assertThat(ask("{\"query\":\" \",\"documentId\":\"a\"}").getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(CALLS.get()).isZero();
    }

    /** 畸形 JSON 返回固定错误且不调用上游。 */
    @Test
    void rejectsMalformedInput() {
        var response = ask("not-json");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(response.getBody().get("code").asText()).isEqualTo("invalid_request");
        assertThat(CALLS.get()).isZero();
    }

    /** 未定义的 API 密钥等字段不可静默进入业务接口。 */
    @Test
    void rejectsUnknownFields() {
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\",\"apiKey\":\"unused\"}")
                .getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(CALLS.get()).isZero();
    }

    /** 超长输入在调用上游前拒绝。 */
    @Test
    void rejectsOversizeQuery() {
        assertThat(ask("{\"query\":\"" + "x".repeat(4001) + "\",\"documentId\":\"a\"}")
                .getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(CALLS.get()).isZero();
    }

    /** 上游失败不重试、不泄漏上游响应正文。 */
    @Test
    void sanitizesUpstreamError() {
        upstreamStatus = 500;
        upstreamBody = "secret upstream response";
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_GATEWAY);
        assertThat(response.getBody().toString()).doesNotContain("secret");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 无效上游 JSON 不被透传给业务调用方。 */
    @Test
    void rejectsMalformedUpstream() {
        upstreamBody = "not-json";
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\"}").getStatusCode())
                .isEqualTo(HttpStatus.BAD_GATEWAY);
    }

    /** 响应归属别的论文时拒绝。 */
    @Test
    void rejectsWrongDocument() {
        upstreamBody = upstreamBody.replace("\"document_id\":\"a\"", "\"document_id\":\"b\"");
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\"}").getStatusCode())
                .isEqualTo(HttpStatus.BAD_GATEWAY);
    }

    /** 不把任意模型声明当成语义支持验证。 */
    @Test
    void rejectsSemanticUpgrade() {
        upstreamBody = upstreamBody.replace("not_verified", "passed");
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\"}").getStatusCode())
                .isEqualTo(HttpStatus.BAD_GATEWAY);
    }

    /** 不完整结论不得以 answered 状态流出。 */
    @Test
    void rejectsInconsistentAnswer() throws Exception {
        ObjectNode body = (ObjectNode) JSON.readTree(upstreamBody);
        ((ObjectNode) body.path("result")).putArray("claims");
        upstreamBody = body.toString();
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\"}").getStatusCode())
                .isEqualTo(HttpStatus.BAD_GATEWAY);
    }

    /** 保留 partial 和缺失信息，不升级为完整回答。 */
    @Test
    void preservesPartialAnswer() throws Exception {
        ObjectNode body = (ObjectNode) JSON.readTree(upstreamBody);
        ((ObjectNode) body.path("result")).put("status", "partial");
        ((ObjectNode) body.path("result")).putArray("missing_info").add("缺少训练时长");
        upstreamBody = body.toString();
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().at("/result/status").asText()).isEqualTo("partial");
        assertThat(response.getBody().at("/result/missingInfo/0").asText()).isEqualTo("缺少训练时长");
    }

    /** 保留无检索证据时的拒答与未适用引用校验状态。 */
    @Test
    void preservesInsufficientAnswer() throws Exception {
        ObjectNode body = (ObjectNode) JSON.readTree(upstreamBody);
        ObjectNode result = (ObjectNode) body.path("result");
        result.put("status", "insufficient");
        result.putArray("claims");
        result.putArray("missing_info").add("未检索到候选证据");
        ((ObjectNode) result.path("validation")).put("citation_identity_and_quote", "not_applicable")
                .put("status_basis", "no_retrieved_evidence");
        upstreamBody = body.toString();
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().at("/result/status").asText()).isEqualTo("insufficient");
        assertThat(response.getBody().at("/result/claims").isEmpty()).isTrue();
    }

    /** 合法错误分类与已知用量通过业务层传递，上游message不外泄。 */
    @ParameterizedTest
    @CsvSource({"document_not_found,404", "invalid_input,400", "context_too_large,422",
            "retrieval_failed,502", "generation_failed,502", "citation_invalid,502",
            "answer_contract_invalid,502", "service_busy,503"})
    void preservesSafeErrorClassification(String code, int status) {
        upstreamStatus = status;
        upstreamBody = "{\"ok\":false,\"error\":{\"code\":\"" + code
                + "\",\"message\":\"secret\",\"service_version\":\"papertrace-internal-question-v1\","
                + "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12}}}";
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode().value()).isEqualTo(status);
        assertThat(response.getBody().path("code").asText()).isEqualTo(code);
        assertThat(response.getBody().at("/usage/totalTokens").asLong()).isEqualTo(12);
        assertThat(response.getBody().toString()).doesNotContain("secret");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 伪造错误码或HTTP状态不匹配时不透传。 */
    @Test
    void rejectsUnknownErrorCode() {
        upstreamStatus = 500;
        upstreamBody = "{\"ok\":false,\"error\":{\"code\":\"secret\",\"service_version\":\"papertrace-internal-question-v1\"}}";
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_GATEWAY);
        assertThat(response.getBody().path("code").asText()).isEqualTo("internal_service_unavailable");
    }

    /** 成功响应只公开经过校验的token统计，不包含生成器其它元数据。 */
    @Test
    void projectsGenerationUsage() throws Exception {
        ObjectNode body = (ObjectNode) JSON.readTree(upstreamBody);
        ((ObjectNode) body.at("/result/generation")).putObject("usage")
                .put("prompt_tokens", 100).put("completion_tokens", 20).put("total_tokens", 120);
        upstreamBody = body.toString();
        var response = ask("{\"query\":\"attention\",\"documentId\":\"a\"}");
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().at("/result/usage/totalTokens").asLong()).isEqualTo(120);
        assertThat(response.getBody().toString()).doesNotContain("private");
    }

    /** 无效用量不被静默转换成有效数字。 */
    @Test
    void rejectsInvalidUsage() throws Exception {
        ObjectNode body = (ObjectNode) JSON.readTree(upstreamBody);
        ((ObjectNode) body.at("/result/generation")).putObject("usage")
                .put("prompt_tokens", 100).put("completion_tokens", 20).put("total_tokens", 999);
        upstreamBody = body.toString();
        assertThat(ask("{\"query\":\"attention\",\"documentId\":\"a\"}").getStatusCode())
                .isEqualTo(HttpStatus.BAD_GATEWAY);
    }
}
