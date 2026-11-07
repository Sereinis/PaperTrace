package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.BusinessExceptionHandler;
import com.papertrace.business.api.UploadController;
import com.papertrace.business.service.UploadService;
import com.papertrace.internal.PaperTracePythonClient;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/** 通过MVC multipart绑定验证上传API和固定安全错误。 */
class UploadApiTest {
    /** 文件以multipart接收，文件名不参与路径，返回202任务，GET不触发新请求。 */
    @Test
    void acceptsMultipartAndListsTask(@TempDir Path root) throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.importDocument(anyString())).thenThrow(new java.io.IOException("secret upstream"));
        var service = new UploadService(root.toString(), client, new ObjectMapper());
        var mvc = MockMvcBuilders.standaloneSetup(new UploadController(service))
                .setControllerAdvice(new BusinessExceptionHandler()).build();
        try {
            var response = mvc.perform(multipart("/api/uploads").file(new MockMultipartFile("file", "../../paper.pdf",
                "application/pdf", "%PDF-1.7\nfixture".getBytes())))
                .andExpect(status().isAccepted()).andReturn().getResponse().getContentAsString();
            assertThat(response).contains("paper.pdf", "pdf_").doesNotContain("../", "secret");
            mvc.perform(get("/api/uploads")).andExpect(status().isOk());
            mvc.perform(multipart("/api/uploads")).andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.code").value("invalid_upload"));
        } finally { service.close(); }
    }

    /** 没有配置库目录时明确返回503，不接受文件。 */
    @Test
    void disabledIsSafe503() throws Exception {
        var service = new UploadService("", mock(PaperTracePythonClient.class), new ObjectMapper());
        var mvc = MockMvcBuilders.standaloneSetup(new UploadController(service))
                .setControllerAdvice(new BusinessExceptionHandler()).build();
        try {
            mvc.perform(multipart("/api/uploads").file(new MockMultipartFile("file", "p.pdf", "application/pdf", "%PDF-x".getBytes())))
                    .andExpect(status().isServiceUnavailable()).andExpect(jsonPath("$.code").value("uploads_disabled"));
        } finally { service.close(); }
    }
}
