package com.papertrace.business;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/** PaperTrace Java 业务后端启动入口。 */
@SpringBootApplication
public class PaperTraceApplication {
    /** 启动 Spring Boot 应用。 */
    public static void main(String[] args) {
        SpringApplication.run(PaperTraceApplication.class, args);
    }
}
