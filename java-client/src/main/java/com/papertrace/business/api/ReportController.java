package com.papertrace.business.api;

import com.papertrace.business.service.ReportApplicationService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/** 提交和读取本机后台报告任务，GET不触发计费。 */
@RestController
@RequestMapping("/api/reports")
public class ReportController {
    private final ReportApplicationService service;
    /** 注入报告应用服务。 */
    public ReportController(ReportApplicationService service) { this.service = service; }
    /** 返回已持久化的后台任务，失败不自动重提。 */
    @PostMapping
    public ResponseEntity<ReportResponse> submit(@Valid @RequestBody ReportRequest request) {
        return ResponseEntity.accepted().body(service.submit(request));
    }
    /** 返回任务进度与已完成报告的安全投影。 */
    @GetMapping
    public List<ReportResponse> list() { return service.list(); }
}
