package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.service.DocumentApplicationService;
import com.papertrace.business.service.QuestionApplicationService.QuestionServiceException;
import com.papertrace.internal.PaperTracePythonClient;
import java.net.http.HttpTimeoutException;
import org.junit.jupiter.api.Test;

/** 验证目录读取的超时与中断边界。 */
class DocumentApplicationServiceTest {
    /** 目录超时只请求一次并返回504。 */
    @Test
    void timeoutDoesNotRetry() throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.documents()).thenThrow(new HttpTimeoutException("private"));
        var service = new DocumentApplicationService(client, new ObjectMapper());
        assertThatThrownBy(service::list).isInstanceOfSatisfying(QuestionServiceException.class,
                error -> assertThat(error.httpStatus()).isEqualTo(504));
        verify(client, times(1)).documents();
    }

    /** 中断后恢复线程标志。 */
    @Test
    void restoresInterrupt() throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.documents()).thenThrow(new InterruptedException("private"));
        try {
            assertThatThrownBy(new DocumentApplicationService(client, new ObjectMapper())::list)
                    .isInstanceOf(QuestionServiceException.class);
            assertThat(Thread.currentThread().isInterrupted()).isTrue();
        } finally { Thread.interrupted(); }
    }
}
