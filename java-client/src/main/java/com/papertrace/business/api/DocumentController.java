package com.papertrace.business.api;

import com.papertrace.business.service.DocumentApplicationService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/** 提供不触发模型请求的已加载论文目录。 */
@RestController
public class DocumentController {
    private final DocumentApplicationService service;

    /** 注入目录应用服务。 */
    public DocumentController(DocumentApplicationService service) { this.service = service; }

    /** 返回已校验的论文目录。 */
    @GetMapping("/api/documents")
    public DocumentResponse list() { return service.list(); }
}
