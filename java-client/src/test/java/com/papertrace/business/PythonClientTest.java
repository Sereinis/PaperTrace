package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;

import com.papertrace.internal.PaperTracePythonClient;
import java.time.Duration;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** 验证内部客户端地址配置边界。 */
class PythonClientTest {
    /** 禁止公网地址、凭据、查询参数和其他协议。 */
    @ParameterizedTest
    @ValueSource(strings = {"https://127.0.0.1", "http://example.com", "http://0.0.0.0",
            "http://user:pass@localhost", "http://localhost?q=1", "http://localhost#fragment"})
    void rejectsExternalEndpoint(String endpoint) {
        assertThatIllegalArgumentException().isThrownBy(() -> new PaperTracePythonClient(endpoint));
    }

    /** 接受显式本机 HTTP 配置。 */
    @ParameterizedTest
    @ValueSource(strings = {"http://127.0.0.1:8765", "http://localhost:8765"})
    void acceptsLocalEndpoint(String endpoint) {
        assertThatCode(() -> new PaperTracePythonClient(endpoint, Duration.ofSeconds(10)))
                .doesNotThrowAnyException();
    }
}
