package com.papertrace.business.api;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.util.List;

/** 显式报告范围与幂等身份，不接受路径、API密钥或生成参数。 */
public record ReportRequest(
        @NotBlank @Pattern(regexp="[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}") String requestId,
        @NotBlank @Size(max=256) String documentId,
        @NotBlank @Size(max=1000) String experimentScope,
        @NotNull @Valid Target resultTarget) {
    /** 固定目标身份和需核对的评测条件，输入不作为论文证据。 */
    public record Target(@NotBlank @Size(max=256) String model,
                         @NotBlank @Size(max=256) String task,
                         @NotBlank @Size(max=256) String dataset,
                         @NotBlank @Size(max=256) String split,
                         @NotBlank @Pattern(regexp="single|ensemble") String modelType,
                         @NotNull @Size(max=4) List<@NotNull @Pattern(regexp="model_selection|checkpoint_averaging|decoding|evaluation_protocol") String> requiredConditions) { }
}
