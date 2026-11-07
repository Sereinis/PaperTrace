package com.papertrace.business.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.DocumentResponse;
import com.papertrace.business.api.DocumentResponse.Document;
import com.papertrace.business.service.QuestionApplicationService.QuestionServiceException;
import com.papertrace.internal.PaperTracePythonClient;
import java.io.IOException;
import java.net.http.HttpTimeoutException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.springframework.stereotype.Service;

/** 校验内部论文目录，再投影为页面可用的最小元数据。 */
@Service
public class DocumentApplicationService {
    private final PaperTracePythonClient client;
    private final ObjectMapper mapper;

    /** 注入本机客户端与JSON解析器。 */
    public DocumentApplicationService(PaperTracePythonClient client, ObjectMapper mapper) {
        this.client = client;
        this.mapper = mapper;
    }

    /** 读取已加载目录；失败不返回缓存旧目录，也不触发模型。 */
    public DocumentResponse list() {
        try {
            JsonNode root = mapper.readTree(client.documents());
            if (root == null || !root.path("ok").isBoolean() || !root.path("ok").booleanValue()
                    || !root.path("service_version").isTextual()
                    || !root.path("service_version").textValue().equals("papertrace-internal-question-v1")
                    || !root.path("documents").isArray()) {
                throw new IllegalStateException();
            }
            var identities = new HashSet<String>();
            var documents = new ArrayList<Document>();
            for (JsonNode row : root.path("documents")) {
                JsonNode id = row.path("document_id"), count = row.path("chunk_count");
                if (!id.isTextual() || id.textValue().isBlank() || id.textValue().length() > 256
                        || !identities.add(id.textValue()) || !count.isIntegralNumber()
                        || !count.canConvertToLong() || count.longValue() < 1) {
                    throw new IllegalStateException();
                }
                documents.add(new Document(id.textValue(), count.longValue()));
            }
            return new DocumentResponse(List.copyOf(documents));
        } catch (HttpTimeoutException error) {
            throw new QuestionServiceException("internal_timeout", 504, null, error);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new QuestionServiceException("论文目录读取已中断", error);
        } catch (IOException | RuntimeException error) {
            throw new QuestionServiceException("论文目录暂不可用", error);
        }
    }
}
