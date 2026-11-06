package com.papertrace.business.api;

/** Java 业务层对外暴露的脱敏错误结构。 */
public record BusinessErrorResponse(String code, String message) {
}
