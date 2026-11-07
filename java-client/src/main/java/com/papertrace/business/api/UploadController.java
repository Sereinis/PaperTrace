package com.papertrace.business.api;

import com.papertrace.business.service.UploadService;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/** 本地单用户PDF上传与任务状态，不接收任意路径。 */
@RestController
@RequestMapping("/api/uploads")
public class UploadController {
    private final UploadService service;
    /** 注入上传任务服务。 */
    public UploadController(UploadService service) { this.service = service; }
    /** 上传后立即返回任务，处理在后台执行。 */
    @PostMapping(consumes = "multipart/form-data")
    public ResponseEntity<UploadResponse> upload(@RequestParam("file") MultipartFile file) {
        return ResponseEntity.accepted().body(service.submit(file));
    }
    /** 读取所有持久化任务，不触发模型或重跑。 */
    @GetMapping
    public List<UploadResponse> list() { return service.list(); }
}
