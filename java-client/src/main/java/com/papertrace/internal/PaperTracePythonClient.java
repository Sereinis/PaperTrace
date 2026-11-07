package com.papertrace.internal;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** 供 Java 业务后端调用本机 PaperTrace Python 内部服务的最小客户端。 */
public final class PaperTracePythonClient {
    private final URI baseUri;
    private final HttpClient client;
    private final Duration timeout;

    /** 仅接受本机 HTTP 地址，避免把问题转发到任意外部 URL。 */
    public PaperTracePythonClient(String endpoint) {
        this(endpoint, Duration.ofSeconds(120));
    }

    /** 创建带有限请求超时的本机客户端。 */
    public PaperTracePythonClient(String endpoint, Duration timeout) {
        if (endpoint == null || timeout == null || timeout.isZero() || timeout.isNegative()
                || timeout.compareTo(Duration.ofSeconds(300)) > 0) {
            throw new IllegalArgumentException("endpoint 和 timeout 必须有效");
        }
        URI uri = URI.create(endpoint);
        String host = uri.getHost();
        if (!"http".equalsIgnoreCase(uri.getScheme())
                || !("127.0.0.1".equalsIgnoreCase(host) || "localhost".equalsIgnoreCase(host))
                || uri.getUserInfo() != null || uri.getQuery() != null || uri.getFragment() != null) {
            throw new IllegalArgumentException("仅允许本机 HTTP endpoint");
        }
        this.baseUri = uri;
        this.timeout = timeout;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3))
                .followRedirects(HttpClient.Redirect.NEVER).build();
    }

    /** 调用单问题接口并返回已校验 HTTP 状态的原始 JSON。 */
    public String ask(String query, String documentId) throws IOException, InterruptedException {
        if (query == null || query.isBlank() || documentId == null || documentId.isBlank()) {
            throw new IllegalArgumentException("query 和 documentId 必须非空");
        }
        String body = "{\"query\":\"" + escape(query) + "\",\"document_id\":\""
                + escape(documentId) + "\"}";
        HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("/v1/question"))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new PythonResponseException(response.statusCode(), response.body());
        }
        return response.body();
    }

    /** 读取本机论文目录，不触发检索或生成模型请求。 */
    public String documents() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder(baseUri.resolve("/v1/documents"))
                .timeout(timeout).GET().build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new PythonResponseException(response.statusCode(), response.body());
        }
        return response.body();
    }

    /** 仅提交哈希身份，不允许传入任意路径，后台导入不自动重试。 */
    public String importDocument(String documentId) throws IOException, InterruptedException {
        if (documentId == null || !documentId.matches("pdf_[0-9a-f]{64}")) {
            throw new IllegalArgumentException("导入身份无效");
        }
        var request = HttpRequest.newBuilder(baseUri.resolve("/v1/import"))
                .timeout(Duration.ofSeconds(900)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("{\"document_id\":\"" + documentId + "\"}")).build();
        var response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) throw new PythonResponseException(response.statusCode(), response.body());
        return response.body();
    }

    /** 保存待业务层白名单解析的响应，不将正文拼入异常消息。 */
    public static final class PythonResponseException extends IOException {
        private final int status;
        private final String body;

        /** 保存内部HTTP状态及响应正文，仅供脱敏分类。 */
        public PythonResponseException(int status, String body) {
            super("Python 内部服务返回 HTTP " + status);
            this.status = status;
            this.body = body;
        }

        /** 返回内部HTTP状态。 */
        public int status() { return status; }

        /** 返回待校验正文，禁止作为对外错误消息。 */
        public String body() { return body; }
    }

    /** 转义 JSON 字符串中的控制字符和反斜杠。 */
    private static String escape(String value) {
        StringBuilder result = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\' -> result.append("\\\\");
                case '"' -> result.append("\\\"");
                case '\r' -> result.append("\\r");
                case '\n' -> result.append("\\n");
                case '\t' -> result.append("\\t");
                default -> {
                    if (c < 0x20) {
                        result.append(String.format("\\u%04x", (int) c));
                    } else {
                        result.append(c);
                    }
                }
            }
        }
        return result.toString();
    }
}
