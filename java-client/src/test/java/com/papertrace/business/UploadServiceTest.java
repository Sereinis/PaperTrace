package com.papertrace.business;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.UploadResponse;
import com.papertrace.business.service.UploadService;
import com.papertrace.internal.PaperTracePythonClient;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.mock.web.MockMultipartFile;

/** 验证本地PDF任务的大小、身份、脱敏失败和单次上游调用。 */
class UploadServiceTest {
    private static final String PDF = "%PDF-1.7\nfixture";

    /** 未配置目录时不能上传，也不触发模型。 */
    @Test
    void disabledWithoutDirectory() throws Exception {
        var client = mock(PaperTracePythonClient.class);
        var service = new UploadService("", client, new ObjectMapper());
        try {
            assertThatThrownBy(() -> service.submit(new MockMultipartFile("file", "p.pdf", "application/pdf", PDF.getBytes())))
                    .isInstanceOfSatisfying(UploadService.UploadException.class, e -> assertThat(e.status).isEqualTo(503));
            verifyNoInteractions(client);
        } finally { service.close(); }
    }

    /** 无效扩展名、签名和超限文件都在发送请求前拒绝。 */
    @Test
    void rejectsInvalidFileWithoutCallingPython(@TempDir Path root) throws Exception {
        var client = mock(PaperTracePythonClient.class);
        var service = new UploadService(root.toString(), client, new ObjectMapper());
        try {
            assertThatThrownBy(() -> service.submit(new MockMultipartFile("file", "x.txt", "text/plain", PDF.getBytes())))
                    .isInstanceOfSatisfying(UploadService.UploadException.class,
                            error -> assertThat(error.code).isEqualTo("invalid_upload"));
            assertThatThrownBy(() -> service.submit(new MockMultipartFile("file", "x.pdf", "application/pdf", "invalid".getBytes())))
                    .isInstanceOf(UploadService.UploadException.class);
            assertThatThrownBy(() -> service.submit(new MockMultipartFile("file", "x.pdf", "application/pdf", new byte[10 * 1024 * 1024 + 1])))
                    .isInstanceOf(UploadService.UploadException.class);
            verifyNoInteractions(client);
        } finally { service.close(); }
    }

    /** 成功后复用相同内容的任务，文件路径片段只影响展示名。 */
    @Test
    void persistsQueuedThenReadyAndReusesDuplicate(@TempDir Path root) throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.importDocument(anyString())).thenAnswer(invocation ->
                "{\"ok\":true,\"service_version\":\"papertrace-internal-question-v1\",\"document\":{\"document_id\":\""
                        + invocation.getArgument(0) + "\",\"chunk_count\":2}}");
        var service = new UploadService(root.toString(), client, new ObjectMapper());
        try {
            var file = new MockMultipartFile("file", "folder\\paper.pdf", "application/pdf", PDF.getBytes());
            UploadResponse first = service.submit(file);
            assertThat(first.status()).isEqualTo("queued");
            UploadResponse ready = await(service, first.documentId(), "ready");
            assertThat(ready.chunkCount()).isEqualTo(2);
            assertThat(ready.fileName()).isEqualTo("paper.pdf");
            assertThat(service.submit(file)).isEqualTo(ready);
            verify(client, times(1)).importDocument(first.documentId());
            assertThat(root.resolve(first.documentId()).resolve("source.pdf")).exists();
        } finally { service.close(); }
    }

    /** 上游失败不泄露正文，重复提交失败文件不新增计费尝试。 */
    @Test
    void upstreamFailureIsSafeAndNotRetried(@TempDir Path root) throws Exception {
        var client = mock(PaperTracePythonClient.class);
        when(client.importDocument(anyString())).thenThrow(
                new PaperTracePythonClient.PythonResponseException(422,
                        "{\"ok\":false,\"error\":{\"service_version\":\"papertrace-internal-question-v1\",\"code\":\"encrypted_pdf\",\"message\":\"secret\"}}"));
        var service = new UploadService(root.toString(), client, new ObjectMapper());
        try {
            UploadResponse first = service.submit(new MockMultipartFile("file", "paper.pdf", "application/pdf", PDF.getBytes()));
            UploadResponse failed = await(service, first.documentId(), "failed");
            assertThat(failed.errorCode()).isEqualTo("encrypted_pdf");
            verify(client, times(1)).importDocument(first.documentId());
            assertThat(service.submit(new MockMultipartFile("file", "paper.pdf", "application/pdf", PDF.getBytes())))
                    .isEqualTo(failed);
        } finally { service.close(); }
    }

    /** 重启将未完成任务标记中断，不重新请求Python，ready与failed保留。 */
    @Test
    void restartNeverResumes(@TempDir Path root) throws Exception {
        var mapper = new ObjectMapper();
        var client = mock(PaperTracePythonClient.class);
        String[] statuses = {"queued", "processing", "ready", "failed"};
        for (int i = 0; i < statuses.length; i++) {
            String id = "pdf_" + Integer.toString(i).repeat(64);
            Path folder = Files.createDirectory(root.resolve(id));
            mapper.writeValue(folder.resolve("task.json").toFile(), new UploadResponse(id, "paper.pdf", statuses[i], null,
                    statuses[i].equals("ready") ? 1 : 0));
        }
        var service = new UploadService(root.toString(), client, mapper);
        try {
            assertThat(service.list()).extracting(UploadResponse::status).containsExactly("failed", "failed", "ready", "failed");
            assertThat(service.list().getFirst().errorCode()).isEqualTo("interrupted");
            verifyNoInteractions(client);
        } finally { service.close(); }
    }

    /** 忙时拒绝另一份新文件，等待超时仍只尝试一次。 */
    @Test
    void busyAndTimeoutAreBounded(@TempDir Path root) throws Exception {
        var client = mock(PaperTracePythonClient.class);
        var started = new java.util.concurrent.CountDownLatch(1);
        var release = new java.util.concurrent.CountDownLatch(1);
        when(client.importDocument(anyString())).thenAnswer(invocation -> {
            started.countDown(); release.await(3, TimeUnit.SECONDS);
            throw new java.net.http.HttpTimeoutException("secret");
        });
        var service = new UploadService(root.toString(), client, new ObjectMapper());
        try {
            var task = service.submit(new MockMultipartFile("file", "p.pdf", "application/pdf", PDF.getBytes()));
            assertThat(started.await(2, TimeUnit.SECONDS)).isTrue();
            assertThatThrownBy(() -> service.submit(new MockMultipartFile("file", "other.pdf", "application/pdf", (PDF + "other").getBytes())))
                    .isInstanceOfSatisfying(UploadService.UploadException.class, e -> assertThat(e.code).isEqualTo("import_busy"));
            release.countDown();
            assertThat(await(service, task.documentId(), "failed").errorCode()).isEqualTo("import_timeout");
            verify(client, times(1)).importDocument(anyString());
        } finally { release.countDown(); service.close(); }
    }

    /** 有界等待后台任务，不等待真实网络。 */
    private static UploadResponse await(UploadService service, String id, String status) throws InterruptedException {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        UploadResponse current;
        do {
            current = service.list().stream().filter(row -> row.documentId().equals(id)).findFirst().orElseThrow();
            if (current.status().equals(status)) return current;
            Thread.sleep(10);
        } while (System.nanoTime() < end);
        fail("任务未达到状态 " + status + ": " + current.status());
        return current;
    }
}
