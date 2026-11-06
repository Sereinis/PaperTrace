package com.papertrace.business.api;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/** Java 业务接口接收的单问题请求。 */
public record QuestionRequest(
        @NotBlank @Size(max = 4000) String query,
        @NotBlank @Size(max = 256) String documentId) {
}
