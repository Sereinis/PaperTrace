package com.papertrace.business;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

/** 通过实际Java HTTP验证论文目录投影及静态页面。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class DocumentApiTest {
    private static final AtomicInteger CALLS = new AtomicInteger();
    private static volatile String rows, envelope;
    private static volatile int upstreamStatus;
    private static final HttpServer UPSTREAM = upstream();
    @Autowired private TestRestTemplate rest;

    /** 启动仅有目录端点的本机模拟上游。 */
    private static HttpServer upstream() {
        try {
            var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/v1/documents", exchange -> {
                CALLS.incrementAndGet();
                byte[] body = (envelope == null ? "{\"ok\":true,\"service_version\":\"papertrace-internal-question-v1\",\"documents\":" + rows + "}" : envelope)
                        .getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(upstreamStatus, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
            server.start();
            return server;
        } catch (IOException error) { throw new IllegalStateException(error); }
    }

    /** 注入本机模拟上游，不访问真实模型。 */
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("papertrace.internal-service-url", () -> "http://127.0.0.1:" + UPSTREAM.getAddress().getPort());
    }

    /** 恢复合法目录与调用次数。 */
    @BeforeEach
    void reset() {
        CALLS.set(0);
        upstreamStatus = 200;
        envelope = null;
        rows = "[{\"document_id\":\"a\",\"chunk_count\":54,\"source_path\":\"private-path\",\"text\":\"private-text\"}]";
    }

    /** 关闭模拟目录服务。 */
    @AfterAll
    static void close() { UPSTREAM.stop(0); }

    /** 只返回稳定ID和分块数，不透传额外字段。 */
    @Test
    void returnsSafeDocuments() {
        var response = rest.getForEntity("/api/documents", JsonNode.class);
        assertThat(response.getStatusCode().value()).isEqualTo(200);
        assertThat(response.getBody().at("/documents/0/documentId").asText()).isEqualTo("a");
        assertThat(response.getBody().at("/documents/0/chunkCount").asInt()).isEqualTo(54);
        assertThat(response.getBody().toString()).doesNotContain("private", "source_path");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 空目录是可显示的合法结果，不编造论文。 */
    @Test
    void preservesEmptyCatalog() {
        rows = "[]";
        assertThat(rest.getForObject("/api/documents", JsonNode.class).path("documents").isEmpty()).isTrue();
    }

    /** 不能提交到问题接口的过长ID不可出现在可选择目录。 */
    @Test
    void rejectsUnaskableDocumentId() {
        rows = "[{\"document_id\":\"" + "a".repeat(257) + "\",\"chunk_count\":1}]";
        assertThat(rest.getForEntity("/api/documents", JsonNode.class).getStatusCode().value()).isEqualTo(502);
    }

    /** 非法目录字段、类型、数量、重复身份及过长ID均拒绝。 */
    @ParameterizedTest
    @ValueSource(strings = {"null", "{}", "[{\"document_id\":\"\",\"chunk_count\":1}]",
        "[{\"document_id\":\"a\",\"chunk_count\":0}]", "[{\"document_id\":\"a\",\"chunk_count\":1.5}]",
        "[{\"document_id\":\"a\",\"chunk_count\":\"1\"}]",
        "[{\"document_id\":\"a\",\"chunk_count\":1},{\"document_id\":\"a\",\"chunk_count\":2}]"})
    void rejectsInvalidCatalog(String value) {
        rows = value;
        var response = rest.getForEntity("/api/documents", JsonNode.class);
        assertThat(response.getStatusCode().value()).isEqualTo(502);
        assertThat(response.getBody().path("code").asText()).isEqualTo("internal_service_unavailable");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 非200、畸形JSON或不符版本都不作为有效目录返回。 */
    @ParameterizedTest
    @ValueSource(strings = {"null", "secret response", "{\"ok\":true,\"service_version\":\"wrong\",\"documents\":[]}",
        "{\"ok\":false,\"service_version\":\"papertrace-internal-question-v1\",\"documents\":[]}"})
    void rejectsInvalidEnvelope(String value) {
        envelope = value;
        assertThat(rest.getForEntity("/api/documents", JsonNode.class).getStatusCode().value()).isEqualTo(502);
    }

    /** 上游失败正文不泄露，且不自动重试。 */
    @Test
    void rejectsUpstreamFailure() {
        upstreamStatus = 500;
        envelope = "secret response";
        var response = rest.getForEntity("/api/documents", JsonNode.class);
        assertThat(response.getStatusCode().value()).isEqualTo(502);
        assertThat(response.getBody().toString()).doesNotContain("secret");
        assertThat(CALLS.get()).isEqualTo(1);
    }

    /** 首页及资源由Spring提供，不请求上游或模型。 */
    @Test
    void servesWorkspace() {
        var response = rest.getForEntity("/", String.class);
        assertThat(response.getStatusCode().value()).isEqualTo(200);
        assertThat(response.getBody()).contains("论文证据问答", "question-form", "Content-Security-Policy");
        assertThat(rest.getForObject("/app.js", String.class)).contains("textContent", "/api/questions");
        assertThat(rest.getForObject("/app.css", String.class)).contains("@media");
        assertThat(CALLS.get()).isZero();
    }
}
