package com.papertrace.business.api;

import java.util.List;

/** Java 业务接口返回的稳定外层响应，不直接暴露 Python HTTP 原始响应。 */
public record QuestionResponse(String serviceVersion, Answer result) {
    /** 向业务调用方公开答案和校验状态，不透传候选全文或生成器内部元数据。 */
    public record Answer(String query, String documentId, String status, List<Claim> claims,
                         List<String> missingInfo, Validation validation) { }

    /** 表示一个结论及其程序定位后的引用。 */
    public record Claim(String text, List<Citation> citations) { }

    /** 引用的论文、页码、原文摘录与字符偏移来自 Python 问答流程。 */
    public record Citation(String evidenceId, String segmentId, String chunkId, String documentId,
                           int pageNumber, String quote, int quoteStart, int quoteEnd) { }

    /** 保留引用定位校验与尚未完成语义验证的区别。 */
    public record Validation(String citationIdentityAndQuote, String semanticSupport, String statusBasis) { }
}
