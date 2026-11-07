package com.papertrace.business.api;

/** Java 业务层对外暴露的脱敏错误结构。 */
public record BusinessErrorResponse(String code, String message, QuestionResponse.Usage usage) {
    /** 构造无已知用量的固定错误。 */
    public BusinessErrorResponse(String code, String message) { this(code, message, null); }
}
