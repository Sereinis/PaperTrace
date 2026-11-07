package com.papertrace.business.api;

/** 持久化任务的安全投影，文件名只作为显示文本，不参与路径计算。 */
public record UploadResponse(String documentId, String fileName, String status, String errorCode, long chunkCount) { }
