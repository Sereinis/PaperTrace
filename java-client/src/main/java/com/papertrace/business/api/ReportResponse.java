package com.papertrace.business.api;

import java.util.List;

/** 报告任务安全投影，不包含检索候选、审计或模型原始输出。 */
public record ReportResponse(String requestId, String documentId, String experimentScope,
                             String status, int completedFields, String currentField,
                             String errorCode, Generation generation, Result result) {
    /** 合法生成用量与未知调用数，不估算账单。 */
    public record Generation(int generationCalls, int unknownUsageCalls,
                             QuestionResponse.Usage knownUsageSubtotal, QuestionResponse.Usage totalUsage) { }
    /** 七字段结果始终标记待人工核对。 */
    public record Result(List<Section> sections, String semanticSupport, String scopeMatch,
                         String crossFieldConsistency, String exhaustiveness) { }
    /** 复用问答的强类型结论与引用。 */
    public record Section(String field, String label, QuestionResponse.Answer answer) { }
}
