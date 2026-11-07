package com.papertrace.business.config;

import com.papertrace.internal.PaperTracePythonClient;
import java.time.Duration;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** 配置 Java 到 Python 的本机内部服务客户端。 */
@Configuration
public class InternalServiceConfig {
    /** 使用受配置约束的内部地址创建客户端。 */
    @Bean
    public PaperTracePythonClient paperTracePythonClient(
            @Value("${papertrace.internal-service-url:http://127.0.0.1:8765}") String endpoint,
            @Value("${papertrace.internal-timeout-seconds:120}") int timeoutSeconds) {
        return new PaperTracePythonClient(endpoint, Duration.ofSeconds(timeoutSeconds));
    }
}
