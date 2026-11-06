package com.papertrace.business.api;

import com.papertrace.business.service.QuestionApplicationService.QuestionServiceException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/** 将输入错误和内部依赖错误转换为不泄露正文的业务响应。 */
@RestControllerAdvice
public class BusinessExceptionHandler {
    /** 处理请求字段校验失败。 */
    @ExceptionHandler({MethodArgumentNotValidException.class, HttpMessageNotReadableException.class})
    public ResponseEntity<BusinessErrorResponse> invalidRequest(Exception error) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(new BusinessErrorResponse("invalid_request", "请求字段无效"));
    }

    /** 处理 Python 内部服务不可用或响应结构错误。 */
    @ExceptionHandler(QuestionServiceException.class)
    public ResponseEntity<BusinessErrorResponse> internalService(QuestionServiceException error) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(new BusinessErrorResponse("internal_service_unavailable", "问答服务暂不可用"));
    }
}
