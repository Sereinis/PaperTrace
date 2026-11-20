"use strict";
(() => {
  const byId = id => document.getElementById(id);
  const form = byId("question-form"), select = byId("document"), query = byId("query");
  const submit = byId("submit"), refresh = byId("refresh"), result = byId("result");
  const file = byId("pdf-file"), upload = byId("upload"), uploadState = byId("upload-state");
  const badge = byId("badge"), state = byId("request-state"), usage = byId("usage");
  const reportSubmit = byId("report-submit"), reportState = byId("report-state"), reportResult = byId("report-result");
  const reportTasks = byId("report-tasks"), reportRefresh = byId("report-refresh");
  const reportExport = byId("report-export"), reportExportState = byId("report-export-state");
  let exportableReport = null;
  const reportFields = ["research_question", "datasets", "model", "baselines", "training_settings", "evaluation_metrics", "results"];
  const reportLabels = ["研究问题", "数据集", "模型", "对照方法", "训练设置", "评价指标", "实验结果"];
  const reportErrors = {interrupted: "服务重启，任务已中断", field_failed: "字段生成或复核失败", report_failed: "报告执行或保存失败",
    reports_disabled: "未配置报告任务目录", service_busy: "已有问答或报告正在执行", request_conflict: "任务身份冲突",
    report_limit: "已达到50份任务上限，请由维护者归档后重启", invalid_input: "报告输入无效", invalid_request: "报告输入无效",
    document_not_found: "论文不存在", internal_service_unavailable: "报告服务不可用，请检查Python服务"};
  let reportReading = false, reportsAvailable = true;
  let busy = false, uploadBusy = false, reportBusy = false;
  const uploadErrors = {
    uploads_disabled: "未配置本地论文库", invalid_upload: "PDF 无效或超过 10MB",
    encrypted_pdf: "暂不支持加密 PDF", page_limit: "PDF 页数超过 60 页",
    text_limit: "提取文本超过限制", chunk_limit: "分块超过 300 块",
    no_extractable_text: "未提取到文本，暂不支持扫描件", pdf_parse_failed: "PDF 解析失败",
    import_timeout: "等待处理超时，上游可能仍在执行", interrupted: "服务重启或任务中断",
    import_already_attempted: "任务已尝试，不会自动续跑", import_busy: "已有论文正在处理",
    upload_conflict: "文件目录已存在，需要人工检查", upload_storage_failed: "文件保存失败"
  };
  const errors = {
    invalid_request: "请求字段无效，请检查问题和论文。", invalid_input: "请求输入不符合约定。",
    invalid_json: "请求格式无效。", invalid_content_length: "请求长度无效。",
    document_not_found: "这篇论文已不在服务中，请刷新论文列表。",
    context_too_large: "候选证据上下文超过限制，本次未返回答案。",
    service_busy: "服务正在处理另一条问题，请稍后手动提交。",
    retrieval_failed: "检索失败；离线查询缓存未命中也会返回此错误。",
    generation_failed: "模型生成失败，已发送的请求可能计费。",
    citation_invalid: "引用定位校验未通过，本次不展示答案。",
    answer_contract_invalid: "答案结构未通过校验，本次不展示答案。",
    invalid_generation_result: "模型输出无效，本次不展示答案。",
    internal_timeout: "等待内部服务超时；上游可能仍在执行，请勿立即重复提交。",
    internal_service_unavailable: "内部论文服务暂不可用，请检查Python服务。"
  };
  /** 所有模型及论文内容都作为纯文本节点展示，不解释HTML或Markdown。 */
  function node(tag, text, className) {
    const value = document.createElement(tag);
    if (text !== undefined) value.textContent = String(text);
    if (className) value.className = className;
    return value;
  }
  /** 仅展示可安全表示且求和一致的生成用量，未知不写成零。 */
  function showUsage(value) {
    const numbers = value && [value.promptTokens, value.completionTokens, value.totalTokens];
    usage.textContent = numbers && numbers.every(n => Number.isSafeInteger(n) && n >= 0)
      && numbers[0] + numbers[1] === numbers[2]
      ? `生成用量：输入 ${numbers[0]} / 输出 ${numbers[1]} / 总计 ${numbers[2]} tokens（非账单）`
      : "生成用量未知（不代表零消费）。";
  }
  /** 同步页面执行状态，阻止同一页面重复发送问答。 */
  function setBusy(value) {
    busy = value;
    document.body.dataset.loading = String(value);
    document.querySelector(".answer").setAttribute("aria-busy", String(value));
    select.disabled = value || uploadBusy || reportBusy || reportReading || !select.value;
    query.disabled = value || uploadBusy || reportBusy || reportReading;
    refresh.disabled = value || uploadBusy || reportBusy || reportReading;
    submit.disabled = value || uploadBusy || reportBusy || reportReading || !select.value;
    upload.disabled = file.disabled = value || uploadBusy || reportBusy || reportReading;
    reportSubmit.disabled = value || uploadBusy || reportBusy || reportReading || !reportsAvailable || !select.value;
    ["scope", "model", "task", "dataset", "split", "model-type", "conditions"].forEach(key => {
      byId(`report-${key}`).disabled = value || uploadBusy || reportBusy || reportReading || !reportsAvailable;
    });
  }
  /** 严格检查任务身份、进度及七字段结果，不把畸形响应当作完整报告。 */
  function validateReport(task) {
    if (!task || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(task.requestId)
        || typeof task.documentId !== "string" || !task.documentId.trim()
        || typeof task.experimentScope !== "string" || !task.experimentScope.trim()
        || !["queued", "processing", "ready", "failed"].includes(task.status)
        || !Number.isSafeInteger(task.completedFields) || task.completedFields < 0 || task.completedFields > 7
        || (task.currentField !== null && !reportFields.includes(task.currentField))
        || (task.errorCode !== null && !["interrupted", "field_failed", "report_failed"].includes(task.errorCode))
        || ((task.status === "failed") !== (task.errorCode !== null))) throw new Error();
    if ((["queued", "failed"].includes(task.status) && task.currentField !== null)
        || (task.status === "queued" && task.completedFields !== 0)) throw new Error();
    const g = task.generation;
    if (g !== null) {
      const validUsage = v => v && [v.promptTokens, v.completionTokens, v.totalTokens].every(n => Number.isSafeInteger(n) && n >= 0)
        && v.promptTokens + v.completionTokens === v.totalTokens;
      if (!g || !Number.isSafeInteger(g.generationCalls) || g.generationCalls < 0 || g.generationCalls > 14
          || !Number.isSafeInteger(g.unknownUsageCalls) || g.unknownUsageCalls < 0 || g.unknownUsageCalls > g.generationCalls
          || !validUsage(g.knownUsageSubtotal)
          || ((g.unknownUsageCalls > 0 || g.generationCalls === 0) ? g.totalUsage !== null
            : !validUsage(g.totalUsage) || g.totalUsage.promptTokens !== g.knownUsageSubtotal.promptTokens
              || g.totalUsage.completionTokens !== g.knownUsageSubtotal.completionTokens)) throw new Error();
    }
    if (task.status === "ready") {
      if (task.completedFields !== 7 || task.currentField !== null || !task.result || !g
          || !Array.isArray(task.result.sections) || task.result.sections.length !== 7
          || ["semanticSupport", "scopeMatch", "crossFieldConsistency", "exhaustiveness"].some(k => task.result[k] !== "not_verified")) throw new Error();
      task.result.sections.forEach((section, i) => {
        if (section.field !== reportFields[i]) throw new Error();
        validate(section.answer, {query: section.answer?.query, documentId: task.documentId}, true);
      });
    } else if (task.result !== null) throw new Error();
    return task;
  }
  /** 将后台报告任务渲染为进度或待核对字段，不展示未完成半份报告。 */
  function renderReport(task) {
    if (!task) return;
    exportableReport = task.status === "ready" ? task : null;
    reportExport.disabled = !exportableReport;
    const labels = {queued: "已排队", processing: "生成中", ready: "已完成·待核对", failed: "未完成"};
    reportState.textContent = `${labels[task.status] || "未知状态"}：${task.completedFields}/7` +
      (task.currentField ? `，当前字段 ${reportLabels[reportFields.indexOf(task.currentField)]}` : "") +
      (task.errorCode ? `（${reportErrors[task.errorCode]}；不会自动重试）` : "");
    reportResult.replaceChildren();
    reportResult.append(node("p", `论文：${task.documentId}\n实验范围：${task.experimentScope}`, "context"));
    const generation = task.generation;
    if (generation) {
      const total = generation.totalUsage, subtotal = generation.knownUsageSubtotal;
      const valid = v => v && [v.promptTokens, v.completionTokens, v.totalTokens].every(n => Number.isSafeInteger(n) && n >= 0)
        && v.promptTokens + v.completionTokens === v.totalTokens;
      const summary = valid(total) ? `总计 ${total.totalTokens} tokens` : valid(subtotal) ? `已知小计 ${subtotal.totalTokens} tokens（非完整总量）` : "用量未知";
      reportResult.append(node("p", `生成调用 ${generation.generationCalls} 次；${summary}；未知用量调用 ${generation.unknownUsageCalls} 次（非账单，不含向量用量）`, "hint"));
    }
    if (task.status !== "ready" || !task.result) return;
    reportResult.append(node("p", "待核对草稿：语义支持、实验范围匹配、跨字段一致性与完整性均未验证。", "missing"));
    task.result.sections.forEach((section, i) => {
      const block = node("article", undefined, "report-section");
      const statuses = {answered: "模型自评完整", partial: "部分信息", insufficient: "证据不足"};
      block.append(node("h3", `${i + 1}. ${reportLabels[i]} · ${statuses[section.answer.status]}`));
      section.answer.claims.forEach(claim => {
        block.append(node("p", claim.text));
        claim.citations.forEach(citation => {
          block.append(node("blockquote", `第 ${citation.pageNumber} 页：${citation.quote}`));
          block.append(node("p", `证据块：${citation.chunkId}`, "context"));
        });
      });
      section.answer.missingInfo.forEach(missing => block.append(node("p", `待核对：${missing}`, "report-missing")));
      reportResult.append(block);
    });
  }
  /** 读取报告任务列表；轮询只读状态，不重新提交或触发生成。 */
  async function loadReports(id) {
    if (reportReading) return;
    exportableReport = null; reportExport.disabled = true; reportExportState.textContent = "";
    reportReading = true; reportRefresh.disabled = true; setBusy(busy);
    try {
      const response = await fetch("/api/reports", {cache: "no-store"});
      if (!response.ok && (await response.json())?.code === "reports_disabled") {
        reportsAvailable = false; reportBusy = false;
        reportResult.replaceChildren(); reportTasks.replaceChildren();
        reportState.textContent = "实验报告未启用：请为Python服务配置 --library-dir；现有问答仍可使用。";
        return;
      }
      if (!response.ok) throw new Error();
      const tasks = await response.json();
      if (!Array.isArray(tasks) || tasks.length > 50) throw new Error();
      tasks.forEach(validateReport);
      reportsAvailable = true;
      if (new Set(tasks.map(t => t.requestId)).size !== tasks.length) throw new Error();
      const previous = id || reportTasks.value;
      reportTasks.replaceChildren();
      tasks.forEach(t => reportTasks.add(new Option(`${t.experimentScope} · ${t.completedFields}/7`, t.requestId)));
      const task = tasks.find(item => item.requestId === previous) || tasks.at(-1);
      if (id && !tasks.some(t => t.requestId === id)) throw new Error();
      if (task) reportTasks.value = task.requestId;
      reportBusy = tasks.some(t => ["queued", "processing"].includes(t.status));
      if (task) renderReport(task);
      else { reportState.textContent = "尚无报告任务。"; reportResult.replaceChildren(); }
      if (reportBusy) setTimeout(() => loadReports(), 1200);
    } catch { reportResult.replaceChildren(); reportState.textContent = "报告任务读取失败，任务可能仍在执行或已计费；请点刷新报告状态，不要立即重提。"; reportBusy = true; }
    finally { reportReading = false; reportRefresh.disabled = false; setBusy(busy); }
  }
  /** 将任意论文文本放在动态长度代码围栏内，避免导出后被解释为HTML、链接或指令式格式。 */
  function reportTextBlock(value) {
    const text = String(value);
    const longest = (text.match(/`+/g) || []).reduce((max, run) => Math.max(max, run.length), 0);
    const fence = "`".repeat(Math.max(3, longest + 1));
    return `${fence}text\n${text}\n${fence}\n`;
  }
  /** 仅从已校验的公开结果生成完整草稿，不包含私有候选、审计或其它任务。 */
  function reportMarkdown(task) {
    validateReport(task);
    if (task.status !== "ready") throw new Error();
    const lines = ["# PaperTrace 实验报告草稿\n",
      "> 待核对草稿：语义支持、实验范围匹配、跨字段一致性与完整性均未验证。",
      "> answered 仅为模型/结构检查自评；引用定位通过不证明结论正确。页码为PDF页序。\n",
      "## 报告身份\n", reportTextBlock(`任务：${task.requestId}\n论文：${task.documentId}\n实验范围：${task.experimentScope}`)];
    const g = task.generation;
    lines.push("## 生成用量\n", `生成调用：${g.generationCalls}；未知用量调用：${g.unknownUsageCalls}。`);
    const u = g.totalUsage || g.knownUsageSubtotal;
    lines.push(`${g.totalUsage ? "总量" : "已知小计（非完整总量）"}：输入 ${u.promptTokens} / 输出 ${u.completionTokens} / 合计 ${u.totalTokens} tokens。`,
      "非账单，不含向量用量。\n");
    const statuses = {answered: "已回答（自评）", partial: "部分回答", insufficient: "证据不足"};
    task.result.sections.forEach((section, i) => {
      const answer = section.answer;
      lines.push(`## ${i + 1}. ${reportLabels[i]} · ${statuses[answer.status]}\n`);
      answer.claims.forEach((claim, index) => {
        lines.push(`### 结论 ${index + 1}\n`, reportTextBlock(claim.text));
        claim.citations.forEach((c, citationIndex) => {
          lines.push(`引用 ${citationIndex + 1}：PDF 第 ${c.pageNumber} 页\n`,
            reportTextBlock(`论文：${c.documentId}\n分块：${c.chunkId}`), reportTextBlock(c.quote));
        });
      });
      if (answer.missingInfo.length) {
        lines.push("### 待核对信息\n");
        answer.missingInfo.forEach(missing => lines.push(reportTextBlock(missing)));
      }
    });
    return lines.join("\n");
  }
  /** 在浏览器本地下载当前已完成报告，不发送网络请求或触发模型；失败不影响已保存任务。 */
  function exportReport() {
    if (!exportableReport || reportReading) return;
    let url;
    try {
      const content = reportMarkdown(exportableReport);
      url = URL.createObjectURL(new Blob([content], {type: "text/markdown;charset=utf-8"}));
      const link = document.createElement("a");
      link.href = url;
      link.download = `PaperTrace-草稿-${exportableReport.requestId}.md`;
      document.body.append(link);
      try { link.click(); } finally { link.remove(); }
      reportExportState.textContent = "已发起草稿下载，请在浏览器下载列表确认；未调用模型。文件含论文摘录，请妥善保存。";
    } catch {
      reportExportState.textContent = "草稿导出失败，未重新生成报告或调用模型；可再次点击导出。";
    } finally {
      if (url) setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }
  /** 提交固定目标的后台报告，失败不自动重提。 */
  async function submitReport() {
    if (busy || uploadBusy || reportBusy || reportReading || !reportsAvailable || !select.value) return;
    const conditions = byId("report-conditions").value.split(",").map(v => v.trim()).filter(Boolean);
    const allowed = new Set(["model_selection", "checkpoint_averaging", "decoding", "evaluation_protocol"]);
    const target = {model: byId("report-model").value.trim(), task: byId("report-task").value.trim(),
      dataset: byId("report-dataset").value.trim(), split: byId("report-split").value.trim(),
      modelType: byId("report-model-type").value, requiredConditions: conditions};
    if (conditions.some(v => !allowed.has(v)) || new Set(conditions).size !== conditions.length || !byId("report-scope").value.trim()
        || Object.values(target).slice(0, 4).some(v => !v || v.length > 256)
        || byId("report-scope").value.length > 1000) { reportState.textContent = "请填写有效的实验范围、目标身份和条件名称。"; return; }
    if (!globalThis.crypto?.randomUUID) { reportState.textContent = "浏览器不支持安全任务身份，请使用本机最新版浏览器。"; return; }
    const id = globalThis.crypto.randomUUID();
    reportBusy = true; setBusy(busy); reportState.textContent = "正在提交报告任务…";
    try {
      const response = await fetch("/api/reports", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({requestId: id, documentId: select.value, experimentScope: byId("report-scope").value.trim(), resultTarget: target})});
      const data = await response.json();
      if (!response.ok) {
        reportState.textContent = `${reportErrors[data?.code] || "报告提交未确认"}；不会自动重试。请刷新报告状态。`;
        reportBusy = true; return;
      }
      validateReport(data);
      if (data.requestId !== id || data.documentId !== select.value || data.experimentScope !== byId("report-scope").value.trim()) throw new Error();
      renderReport(data); await loadReports(id);
    } catch { reportState.textContent = "提交状态未知，任务可能已启动或计费；不会自动重试，请刷新报告状态。"; reportBusy = true; }
    finally { setBusy(busy); }
  }
  /** 只读取元数据；空目录和网络失败保留禁用态，不自动重试。 */
  async function loadDocuments(preferred) {
    refresh.disabled = true;
    select.disabled = submit.disabled = true;
    const previous = typeof preferred === "string" ? preferred : select.value;
    select.replaceChildren(new Option("正在加载论文…", ""));
    byId("catalog-state").textContent = "正在读取已加载论文…";
    try {
      const response = await fetch("/api/documents", {cache: "no-store"});
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (!Array.isArray(data.documents) || data.documents.some(d => typeof d.documentId !== "string"
          || !d.documentId.trim() || !Number.isSafeInteger(d.chunkCount) || d.chunkCount < 1)) throw new Error();
      select.replaceChildren();
      for (const d of data.documents) select.add(new Option(`${d.documentId} · ${d.chunkCount} 块`, d.documentId));
      if (data.documents.some(d => d.documentId === previous)) select.value = previous;
      if (!data.documents.length) select.add(new Option("暂无已加载论文", ""));
      byId("catalog-state").textContent = data.documents.length
        ? `已加载 ${data.documents.length} 篇论文。` : "目录为空，请先上传 PDF。";
    } catch {
      select.replaceChildren(new Option("论文列表不可用", ""));
      byId("catalog-state").textContent = "论文列表读取失败，请检查Python服务后手动刷新。";
    } finally {
      setBusy(busy);
    }
  }
  /** 读取并校验任务状态，只展示固定错误说明，不自动重试网络错误。 */
  async function loadUploads() {
    const response = await fetch("/api/uploads", {cache: "no-store"});
    if (!response.ok) throw new Error();
    const tasks = await response.json();
    if (!Array.isArray(tasks) || tasks.some(t => !t || typeof t.documentId !== "string"
        || !/^pdf_[0-9a-f]{64}$/.test(t.documentId) || typeof t.fileName !== "string"
        || !["queued", "processing", "ready", "failed"].includes(t.status)
        || !Number.isSafeInteger(t.chunkCount) || t.chunkCount < 0
        || (t.status === "ready" && t.chunkCount < 1))) throw new Error();
    const labels = {queued: "等待处理", processing: "正在解析和向量化", ready: "已就绪", failed: "处理失败"};
    uploadState.textContent = tasks.length ? tasks.map(t => `${t.fileName}：${labels[t.status]}${t.status === "failed"
      ? `（${uploadErrors[t.errorCode] || "处理未完成"}；不会自动重试，相同文件重传仍复用失败状态）` : ""}`).join("\n")
      : "支持文本型 PDF：最多 10MB、60 页、300 块。处理会发送提取文本到向量 API 并可能计费。";
    return tasks;
  }
  /** 页面重载也能继续观察未完成任务；只轮询状态，不重新提交或续跑。 */
  async function observeUploads(target) {
    let hadActive = false;
    for (;;) {
      const tasks = await loadUploads();
      if (target && !tasks.some(t => t.documentId === target)) throw new Error();
      const active = tasks.some(t => ["queued", "processing"].includes(t.status));
      uploadBusy = active;
      setBusy(busy);
      if (!active) {
        if (hadActive || target) await loadDocuments(tasks.find(t => t.documentId === target && t.status === "ready")?.documentId);
        return;
      }
      hadActive = true;
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  }
  /** 上传一次并轮询任务状态；失败不自动重新提交文件。 */
  async function submitUpload() {
    if (busy || uploadBusy || reportBusy || reportReading) return;
    if (!file.files.length) { uploadState.textContent = "请选择一个 PDF 文件。"; return; }
    const selected = file.files[0];
    if (!selected.name.toLowerCase().endsWith(".pdf") || selected.size < 5 || selected.size > 10 * 1024 * 1024) {
      uploadState.textContent = "仅支持不超过 10MB 的非空 PDF。"; return;
    }
    uploadBusy = true; setBusy(busy);
    uploadState.textContent = "正在上传…";
    try {
      const formData = new FormData(); formData.append("file", selected);
      const response = await fetch("/api/uploads", {method: "POST", body: formData});
      const data = await response.json();
      if (!response.ok) { uploadState.textContent = `${uploadErrors[data?.code] || "上传失败"}；不会自动重试。`; return; }
      if (!data || !/^pdf_[0-9a-f]{64}$/.test(data.documentId)) throw new Error();
      await observeUploads(data.documentId);
    } catch { uploadState.textContent = "上传或状态读取失败，不会自动重试；上游可能仍在处理或已计费。请稍后刷新页面查看。"; }
    finally { uploadBusy = false; setBusy(busy); }
  }
  /** 在渲染前检查最小结果契约，避免将异常状态解释为完整回答。 */
  function validate(answer, sent, report = false) {
    if (!answer || answer.query !== sent.query || answer.documentId !== sent.documentId
        || !["answered", "partial", "insufficient"].includes(answer.status)
        || !Array.isArray(answer.claims) || !Array.isArray(answer.missingInfo)
        || answer.missingInfo.some(v => typeof v !== "string")
        || answer.validation?.semanticSupport !== "not_verified"
        || !["passed", "not_applicable"].includes(answer.validation?.citationIdentityAndQuote)
        || !["model_self_assessment", "no_retrieved_evidence", ...(report ? ["generation_and_review_model_assessment",
          "structured_identity_link_checks_and_model_review"] : [])].includes(answer.validation?.statusBasis)) throw new Error();
    if ((answer.status === "answered" && (!answer.claims.length || answer.missingInfo.length))
        || (answer.status === "partial" && (!answer.claims.length || !answer.missingInfo.length))
        || (answer.status === "insufficient" && (answer.claims.length || !answer.missingInfo.length))) throw new Error();
    for (const claim of answer.claims) {
      if (typeof claim.text !== "string" || !Array.isArray(claim.citations) || !claim.citations.length) throw new Error();
      for (const c of claim.citations) {
        if (c.documentId !== sent.documentId || typeof c.quote !== "string" || typeof c.chunkId !== "string"
            || !Number.isSafeInteger(c.pageNumber) || c.pageNumber < 1) throw new Error();
      }
    }
  }
  /** 逐条显示结论与原文证据，并保留缺失项及未验证的语义边界。 */
  function render(answer) {
    const labels = {answered: "完整回答 · 模型自评", partial: "部分回答", insufficient: "证据不足"};
    badge.textContent = labels[answer.status];
    badge.dataset.status = answer.status;
    state.textContent = answer.status === "insufficient" ? "本次未提供结论，请查看缺失信息。" : "已返回答案。请逐条结合引文核对。";
    result.append(node("p", `论文：${answer.documentId}\n问题：${answer.query}`, "context"));
    answer.claims.forEach((claim, i) => {
      const section = node("article", undefined, "claim");
      section.append(node("h3", `${i + 1}. ${claim.text}`));
      for (const c of claim.citations) {
        const citation = node("div", undefined, "citation");
        citation.append(node("p", `第 ${c.pageNumber} 页 · ${c.documentId}`));
        citation.append(node("blockquote", c.quote));
        citation.append(node("p", `证据块：${c.chunkId}`));
        section.append(citation);
      }
      result.append(section);
    });
    if (answer.missingInfo.length) {
      const missing = node("section", undefined, "missing"), list = node("ul");
      missing.append(node("h3", "缺失信息 / 待核对"));
      answer.missingInfo.forEach(text => list.append(node("li", text)));
      missing.append(list);
      result.append(missing);
    }
    result.append(node("p", `引用定位：${answer.validation.citationIdentityAndQuote === "passed" ? "通过" : "不适用"} · 语义支持：未验证 · 回答充分性：${answer.validation.statusBasis === "no_retrieved_evidence" ? "无检索证据" : "模型自评"}`, "validation"));
    showUsage(answer.usage);
  }
  /** 单次发送，不取消后静默重发，也不保存问题/答案到浏览器存储。 */
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || uploadBusy || reportBusy || reportReading || !form.reportValidity() || !query.value.trim()) return;
    const sent = {query: query.value, documentId: select.value};
    result.replaceChildren();
    usage.textContent = "";
    badge.textContent = "处理中";
    badge.dataset.status = "loading";
    state.textContent = "正在检索证据并生成答案，请勿重复提交。关闭页面不代表上游任务取消。";
    setBusy(true);
    try {
      const response = await fetch("/api/questions", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(sent)});
      const data = await response.json();
      if (!response.ok) {
        badge.textContent = "未完成";
        badge.dataset.status = "error";
        state.textContent = `${errors[data.code] || "服务返回未识别的错误，本次不展示答案。"} 不会自动重试；已发送请求可能计费。`;
        showUsage(data.usage);
        return;
      }
      validate(data.result, sent);
      render(data.result);
    } catch {
      result.replaceChildren();
      badge.textContent = "未完成";
      badge.dataset.status = "error";
      state.textContent = "网络中断或响应结构无效，未展示答案。不会自动重试；上游可能仍在执行或已计费。";
      showUsage(null);
    } finally { setBusy(false); }
  });
  refresh.addEventListener("click", () => loadDocuments());
  upload.addEventListener("click", submitUpload);
  reportSubmit.addEventListener("click", submitReport);
  reportRefresh.addEventListener("click", () => loadReports());
  reportTasks.addEventListener("change", () => loadReports(reportTasks.value));
  reportExport.addEventListener("click", exportReport);
  loadDocuments();
  observeUploads().catch(() => {
    uploadBusy = false; setBusy(busy);
    uploadState.textContent = "任务状态不可用，请检查服务后刷新页面；不会自动重新提交任务。";
  });
  loadReports();
})();
