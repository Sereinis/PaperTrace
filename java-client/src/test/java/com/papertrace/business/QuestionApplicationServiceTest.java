package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.service.QuestionApplicationService;
import com.papertrace.internal.PaperTracePythonClient;
import org.junit.jupiter.api.Test;

/** 验证应用编排层的线程中断边界。 */
class QuestionApplicationServiceTest {
    /** 内部请求中断后必须恢复线程标志且不重试。 */
    @Test
    void restoresInterrupt() throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.ask("q", "a")).thenThrow(new InterruptedException("private"));
        try {
            var service = new QuestionApplicationService(client, new ObjectMapper());
            assertThatThrownBy(() -> service.ask("q", "a"))
                    .isInstanceOf(QuestionApplicationService.QuestionServiceException.class)
                    .hasMessage("问答内部服务请求已中断");
            assertThat(Thread.currentThread().isInterrupted()).isTrue();
            verify(client, times(1)).ask("q", "a");
        } finally {
            Thread.interrupted();
        }
    }
}
