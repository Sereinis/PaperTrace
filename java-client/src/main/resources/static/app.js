"use strict";
(() => {
  const byId = id => document.getElementById(id);
  const form = byId("question-form"), select = byId("document"), query = byId("query");
  const submit = byId("submit"), refresh = byId("refresh"), result = byId("result");
  const file = byId("pdf-file"), upload = byId("upload"), uploadState = byId("upload-state");
  const badge = byId("badge"), state = byId("request-state"), usage = byId("usage");
  let busy = false, uploadBusy = false;
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
    select.disabled = value || uploadBusy || !select.value;
    query.disabled = value || uploadBusy;
    refresh.disabled = value || uploadBusy;
    submit.disabled = value || uploadBusy || !select.value;
    upload.disabled = file.disabled = value || uploadBusy;
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
    if (busy || uploadBusy) return;
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
  function validate(answer, sent) {
    if (!answer || answer.query !== sent.query || answer.documentId !== sent.documentId
        || !["answered", "partial", "insufficient"].includes(answer.status)
        || !Array.isArray(answer.claims) || !Array.isArray(answer.missingInfo)
        || answer.missingInfo.some(v => typeof v !== "string")
        || answer.validation?.semanticSupport !== "not_verified"
        || !["passed", "not_applicable"].includes(answer.validation?.citationIdentityAndQuote)
        || !["model_self_assessment", "no_retrieved_evidence"].includes(answer.validation?.statusBasis)) throw new Error();
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
    if (busy || uploadBusy || !form.reportValidity() || !query.value.trim()) return;
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
  loadDocuments();
  observeUploads().catch(() => {
    uploadBusy = false; setBusy(busy);
    uploadState.textContent = "任务状态不可用，请检查服务后刷新页面；不会自动重新提交任务。";
  });
})();
