"use strict";
(() => {
  const byId = id => document.getElementById(id);
  const form = byId("question-form"), select = byId("document"), query = byId("query");
  const submit = byId("submit"), refresh = byId("refresh"), result = byId("result");
  const badge = byId("badge"), state = byId("request-state"), usage = byId("usage");
  let busy = false;
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
    select.disabled = value || !select.value;
    query.disabled = value;
    refresh.disabled = value;
    submit.disabled = value || !select.value;
  }
  /** 只读取元数据；空目录和网络失败保留禁用态，不自动重试。 */
  async function loadDocuments() {
    refresh.disabled = true;
    select.disabled = submit.disabled = true;
    const previous = select.value;
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
        ? `已加载 ${data.documents.length} 篇论文。显示稳定ID，暂不支持上传。` : "目录为空，请在Python服务启动时加载论文。";
    } catch {
      select.replaceChildren(new Option("论文列表不可用", ""));
      byId("catalog-state").textContent = "论文列表读取失败，请检查Python服务后手动刷新。";
    } finally {
      refresh.disabled = false;
      select.disabled = submit.disabled = !select.value;
    }
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
    if (busy || !form.reportValidity() || !query.value.trim()) return;
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
  refresh.addEventListener("click", loadDocuments);
  loadDocuments();
})();
