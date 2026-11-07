package com.papertrace.business.api;

import java.util.List;

/** 已加载论文的安全目录，不包含本地文件路径或候选全文。 */
public record DocumentResponse(List<Document> documents) {
    /** 论文稳定身份与实际加载的分块数量，ID不冒充论文标题。 */
    public record Document(String documentId, long chunkCount) { }
}
