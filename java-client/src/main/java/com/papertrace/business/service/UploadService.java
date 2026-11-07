package com.papertrace.business.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.papertrace.business.api.UploadResponse;
import com.papertrace.internal.PaperTracePythonClient;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/** 单用户本地上传任务：先原子保存，再执行一次导入，不自动重试。 */
@Service
public class UploadService {
    private final Path root;
    private final PaperTracePythonClient client;
    private final ObjectMapper mapper;
    private final Map<String, UploadResponse> tasks = new TreeMap<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private boolean active;

    /** 必须显式配置与Python相同的绝对目录；重启将未完成任务标记失败。 */
    public UploadService(@Value("${papertrace.library-dir:}") String directory,
                         PaperTracePythonClient client, ObjectMapper mapper) throws IOException {
        this.client = client;
        this.mapper = mapper;
        if (directory.isBlank()) { root = null; return; }
        Path configured = Path.of(directory);
        if (!configured.isAbsolute()) throw new IllegalArgumentException("论文库必须为绝对路径");
        Files.createDirectories(configured);
        root = configured.toRealPath();
        try (var folders = Files.newDirectoryStream(root, "pdf_*")) {
            for (Path folder : folders) {
                if (!folder.getFileName().toString().matches("pdf_[0-9a-f]{64}") || Files.isSymbolicLink(folder)
                        || !folder.toRealPath().getParent().equals(root))
                    throw new IOException("论文库身份无效");
                Path metadata = folder.resolve("task.json");
                if (!Files.exists(metadata)) continue;
                if (Files.isSymbolicLink(metadata)) throw new IOException("任务不可为符号链接");
                UploadResponse task = mapper.readValue(metadata.toFile(), UploadResponse.class);
                if (!folder.getFileName().toString().equals(task.documentId())
                        || !Set.of("queued", "processing", "ready", "failed").contains(task.status()))
                    throw new IOException("任务元数据无效");
                if (Set.of("queued", "processing").contains(task.status())) {
                    task = new UploadResponse(task.documentId(), task.fileName(), "failed", "interrupted", 0);
                    persist(task);
                }
                tasks.put(task.documentId(), task);
            }
        }
    }

    /** 返回安全状态副本，任务失败不隐藏，未启用时返回空列表。 */
    public synchronized List<UploadResponse> list() { return List.copyOf(tasks.values()); }

    /** 校验扩展名、字节和PDF签名，以内容哈希寻址，重复上传复用状态。 */
    public synchronized UploadResponse submit(MultipartFile file) {
        if (root == null) throw new UploadException("uploads_disabled", 503);
        String name = file.getOriginalFilename();
        if (name == null || name.length() > 180 || name.chars().anyMatch(c -> c < 32)
                || !name.toLowerCase(Locale.ROOT).endsWith(".pdf") || file.isEmpty()
                || file.getSize() > 10 * 1024 * 1024) throw new UploadException("invalid_upload", 400);
        // 文件名仅展示，剥离目录部分，不作为任何落盘路径。
        name = name.replace('\\', '/');
        name = name.substring(name.lastIndexOf('/') + 1);
        try {
            byte[] bytes;
            try (var input = file.getInputStream()) { bytes = input.readNBytes(10 * 1024 * 1024 + 1); }
            if (bytes.length < 5 || bytes.length > 10 * 1024 * 1024
                    || !new String(bytes, 0, 5, java.nio.charset.StandardCharsets.US_ASCII).equals("%PDF-"))
                throw new UploadException("invalid_upload", 400);
            String id = "pdf_" + HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            if (tasks.containsKey(id)) return tasks.get(id);
            if (active) throw new UploadException("import_busy", 409);
            Path folder = root.resolve(id);
            if (Files.exists(folder)) throw new UploadException("upload_conflict", 409);
            Files.createDirectory(folder);
            atomic(folder.resolve("source.pdf"), bytes);
            var task = new UploadResponse(id, name, "queued", null, 0);
            persist(task);
            tasks.put(id, task);
            active = true;
            executor.submit(() -> process(id));
            return task;
        } catch (UploadException error) { throw error;
        } catch (Exception error) { throw new UploadException("upload_storage_failed", 500); }
    }

    /** 后台仅请求一次Python导入；超时或失败保留状态，不续跑。 */
    private void process(String id) {
        try {
            update(id, "processing", null, 0);
            var response = mapper.readTree(client.importDocument(id));
            var document = response.path("document");
            if (!response.path("ok").isBoolean() || !response.path("ok").booleanValue()
                    || !response.path("service_version").asText().equals("papertrace-internal-question-v1")
                    || !document.path("document_id").asText().equals(id)
                    || !document.path("chunk_count").isIntegralNumber()
                    || !document.path("chunk_count").canConvertToLong() || document.path("chunk_count").asLong() < 1
                    || document.path("chunk_count").asLong() > 300)
                throw new IOException("导入响应无效");
            update(id, "ready", null, document.path("chunk_count").asLong());
        } catch (Exception error) {
            String code = "import_failed";
            if (error instanceof java.net.http.HttpTimeoutException) code = "import_timeout";
            if (error instanceof InterruptedException) { Thread.currentThread().interrupt(); code = "interrupted"; }
            if (error instanceof PaperTracePythonClient.PythonResponseException upstream) {
                try {
                    var envelope = mapper.readTree(upstream.body());
                    String candidate = envelope.path("error").path("code").asText();
                    if (upstream.status() == 422 && envelope.path("ok").isBoolean() && !envelope.path("ok").booleanValue()
                            && envelope.path("error").path("service_version").asText().equals("papertrace-internal-question-v1")
                            && Set.of("encrypted_pdf", "page_limit", "text_limit", "chunk_limit", "no_extractable_text",
                               "pdf_parse_failed", "import_already_attempted", "invalid_upload", "uploads_disabled", "import_busy").contains(candidate))
                        code = candidate;
                } catch (Exception ignored) { /* 未知正文不对外暴露。 */ }
            }
            try { update(id, "failed", code, 0); } catch (IOException ignored) { /* 保留内存失败状态，重启恢复处理中断。 */ }
        } finally { synchronized (this) { active = false; } }
    }

    /** 先更新内存以便磁盘失败也能对当前页面报告，再持久化。 */
    private synchronized void update(String id, String status, String code, long count) throws IOException {
        var task = new UploadResponse(id, tasks.get(id).fileName(), status, code, count);
        tasks.put(id, task);
        persist(task);
    }

    /** 用固定元数据文件名原子保存任务状态。 */
    private void persist(UploadResponse task) throws IOException {
        atomic(root.resolve(task.documentId()).resolve("task.json"), mapper.writeValueAsBytes(task));
    }

    /** 同目录临时文件原子替换，不采用原始文件名。 */
    private static void atomic(Path target, byte[] bytes) throws IOException {
        Path temporary = Files.createTempFile(target.getParent(), ".upload-", ".tmp");
        try {
            Files.write(temporary, bytes);
            Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temporary); }
    }

    /** 关闭自身执行器，不等待或自动恢复计费任务。 */
    @PreDestroy
    public void close() { executor.shutdownNow(); }

    /** 固定安全上传错误，不包含原始正文或路径。 */
    public static class UploadException extends RuntimeException {
        public final String code;
        public final int status;
        /** 保存错误分类及HTTP状态。 */
        public UploadException(String code, int status) { super("论文上传未完成"); this.code = code; this.status = status; }
    }
}
