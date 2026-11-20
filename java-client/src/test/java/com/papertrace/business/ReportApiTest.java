package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** 实际Java HTTP到模拟Python端点：报告任务、引用投影及计费安全边界。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ReportApiTest {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final AtomicInteger CALLS = new AtomicInteger();
    private static volatile JsonNode response, received;
    private static volatile String method;
    private static volatile int status;
    private static final String ID = "12345678-1234-4234-8234-123456789abc";
    private static final HttpServer UPSTREAM = upstream();
    @Autowired private TestRestTemplate rest;

    private static HttpServer upstream() {
        try {
            var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/v1/reports", exchange -> {
                CALLS.incrementAndGet(); method = exchange.getRequestMethod();
                byte[] input = exchange.getRequestBody().readAllBytes();
                received = input.length == 0 ? null : MAPPER.readTree(input);
                byte[] body = MAPPER.writeValueAsBytes(response);
                exchange.sendResponseHeaders(status, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            server.start(); return server;
        } catch (Exception error) { throw new IllegalStateException(error); }
    }
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("papertrace.internal-service-url", () -> "http://127.0.0.1:" + UPSTREAM.getAddress().getPort());
    }
    @AfterAll static void close() { UPSTREAM.stop(0); }
    @BeforeEach void reset() { CALLS.set(0); status = 200; received = null; response = envelope(); }

    private static ObjectNode envelope() {
        return MAPPER.createObjectNode().put("ok", true).put("service_version", "papertrace-internal-question-v1");
    }
    private static ObjectNode request() {
        var request = MAPPER.createObjectNode().put("requestId", ID).put("documentId", "a").put("experimentScope", "Alpha on X");
        var target = request.putObject("resultTarget");
        target.put("model", "Alpha").put("task", "classification").put("dataset", "X").put("split", "test").put("modelType", "single");
        target.putArray("requiredConditions");
        return request;
    }
    private static ObjectNode task(boolean ready) throws Exception {
        var task = MAPPER.createObjectNode().put("status", ready ? "ready" : "queued").put("completed_fields", ready ? 7 : 0);
        task.putNull("current_field").putNull("error_code").putNull("generation").putNull("result");
        var request = task.putObject("request");
        request.put("request_id", ID).put("document_id", "a").put("experiment_scope", "Alpha on X");
        var target = request.putObject("result_target");
        target.put("model", "Alpha").put("task", "classification").put("dataset", "X").put("split", "test").put("model_type", "single");
        target.putArray("required_conditions");
        if (ready) {
            var generation = task.putObject("generation").put("generation_calls", 14).put("unknown_usage_calls", 1);
            generation.putObject("known_usage_subtotal").put("prompt_tokens", 10).put("completion_tokens", 2).put("total_tokens", 12);
            generation.putNull("total_usage");
            var result = task.putObject("result").put("document_id", "a").put("experiment_scope", "Alpha on X");
            result.set("generation", generation.deepCopy());
            var validation = result.putObject("validation");
            for (var key : List.of("semantic_support", "experiment_scope_match", "cross_field_consistency", "exhaustiveness")) validation.put(key, "not_verified");
            var sections = result.putArray("sections");
            for (var key : List.of("research_question", "datasets", "model", "baselines", "training_settings", "evaluation_metrics", "results")) {
                var qa = sections.addObject().put("field", key).putObject("qa");
                qa.put("query", "q").put("document_id", "a").put("status", "answered");
                var claim = qa.putArray("claims").addObject().put("text", "<script>claim</script>");
                var citation = claim.putArray("citations").addObject();
                citation.put("evidence_id", "E1").put("segment_id", "E1.S1").put("chunk_id", "a1").put("document_id", "a")
                        .put("page_number", 2).put("quote", "Alpha.").put("quote_start", 0).put("quote_end", 6);
                qa.putArray("missing_info");
                qa.putObject("validation").put("semantic_support", "not_verified").put("citation_identity_and_quote", "passed")
                        .put("status_basis", "generation_and_review_model_assessment");
                qa.putObject("generation").putNull("usage");
                qa.put("evidence", "private-candidate").put("audit", "private-audit");
            }
        }
        return task;
    }
    @Test void submitsOnceWithBoundIdentityAndReadsWithoutPosting() throws Exception {
        ((ObjectNode) response).set("task", task(false)); status = 202;
        var submitted = rest.postForEntity("/api/reports", request(), JsonNode.class);
        assertThat(submitted.getStatusCode().value()).isEqualTo(202);
        assertThat(submitted.getBody().path("requestId").asText()).isEqualTo(ID);
        assertThat(method).isEqualTo("POST");
        assertThat(received.at("/result_target/model_type").asText()).isEqualTo("single");
        assertThat(CALLS.get()).isEqualTo(1);
        status = 200; response = envelope(); ((ObjectNode) response).putArray("tasks").add(task(true));
        var listed = rest.getForEntity("/api/reports", JsonNode.class);
        assertThat(listed.getStatusCode().value()).isEqualTo(200);
        assertThat(method).isEqualTo("GET"); assertThat(received).isNull();
        assertThat(listed.getBody().at("/0/result/sections").size()).isEqualTo(7);
        assertThat(listed.getBody().at("/0/generation/totalUsage").isNull()).isTrue();
        assertThat(listed.getBody().at("/0/result/sections/0/answer/claims/0/citations/0/pageNumber").asInt()).isEqualTo(2);
        assertThat(listed.getBody().toString()).doesNotContain("private-candidate", "private-audit");
    }
    @ParameterizedTest @ValueSource(strings = {"identity", "order", "semantic", "quote", "usage", "partial"})
    void rejectsCorruptUpstream(String kind) throws Exception {
        var task = task(true);
        switch (kind) {
            case "identity" -> ((ObjectNode) task.path("result")).put("document_id", "wrong");
            case "order" -> ((ObjectNode) task.at("/result/sections/0")).put("field", "results");
            case "semantic" -> ((ObjectNode) task.at("/result/validation")).put("semantic_support", "verified");
            case "quote" -> ((ObjectNode) task.at("/result/sections/0/qa/claims/0/citations/0")).put("quote_end", 9);
            case "usage" -> ((ObjectNode) task.path("generation")).put("generation_calls", 15);
            case "partial" -> task.put("status", "processing");
        }
        ((ObjectNode) response).putArray("tasks").add(task);
        var result = rest.getForEntity("/api/reports", JsonNode.class);
        assertThat(result.getStatusCode().value()).isEqualTo(502);
        assertThat(result.getBody().path("code").asText()).isEqualTo("internal_service_unavailable");
        assertThat(CALLS.get()).isEqualTo(1);
    }
    @Test void rejectsInvalidInputsBeforeCallingPython() {
        var request = request(); request.put("requestId", "../escape");
        assertThat(rest.postForEntity("/api/reports", request, JsonNode.class).getStatusCode().value()).isEqualTo(400);
        request = request(); ((ObjectNode) request.path("resultTarget")).putArray("requiredConditions").add("decoding").add("decoding");
        assertThat(rest.postForEntity("/api/reports", request, JsonNode.class).getStatusCode().value()).isEqualTo(400);
        assertThat(CALLS.get()).isZero();
    }
    @Test void safeUpstreamErrorIsNotRetried() {
        status = 503;
        var error = MAPPER.createObjectNode().put("ok", false);
        error.putObject("error").put("code", "service_busy").put("service_version", "papertrace-internal-question-v1").put("message", "secret");
        response = error;
        var result = rest.postForEntity("/api/reports", request(), JsonNode.class);
        assertThat(result.getStatusCode().value()).isEqualTo(503);
        assertThat(result.getBody().path("code").asText()).isEqualTo("service_busy");
        assertThat(result.getBody().toString()).doesNotContain("secret");
        assertThat(CALLS.get()).isEqualTo(1);
    }
}
