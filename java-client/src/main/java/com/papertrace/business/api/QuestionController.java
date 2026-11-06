package com.papertrace.business.api;

import com.papertrace.business.service.QuestionApplicationService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** 对外提供单问题论文问答的 Java 业务接口。 */
@RestController
@RequestMapping("/api/questions")
public class QuestionController {
    private final QuestionApplicationService service;

    /** 注入问答应用服务。 */
    public QuestionController(QuestionApplicationService service) {
        this.service = service;
    }

    /** 校验请求后返回经过业务层解析的问答结果。 */
    @PostMapping
    public ResponseEntity<QuestionResponse> ask(@Valid @RequestBody QuestionRequest request) {
        return ResponseEntity.ok(service.ask(request.query(), request.documentId()));
    }
}
