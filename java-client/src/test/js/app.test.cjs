/** 使用最小DOM替身离线验证页面状态，不操作浏览器或调用真实模型。 */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = readFileSync(path.join(__dirname, '../../main/resources/static/app.js'), 'utf8');

class Element {
  /** 保存纯文本、子节点及事件，不实现HTML解释能力。 */
  constructor(tag = 'div', text = '') {
    this.tag = tag; this.textContent = text; this.children = []; this.dataset = {};
    this.listeners = {}; this.value = ''; this.disabled = false; this.files = [];
  }
  /** 收集子节点。 */
  append(...items) { this.children.push(...items); }
  /** 替换子节点并模拟下拉框默认选中第一项。 */
  replaceChildren(...items) { this.children = items; this.value = items[0]?.value || ''; }
  /** 追加下拉选项。 */
  add(item) { this.children.push(item); if (!this.value) this.value = item.value; }
  /** 收集属性。 */
  setAttribute(key, value) { this[key] = value; }
  /** 保存事件回调。 */
  addEventListener(name, callback) { this.listeners[name] = callback; }
  /** 测试由代码执行的其它状态，不模拟浏览器原生表单校验。 */
  reportValidity() { return true; }
  /** 模拟本地下载点击及临时节点清理。 */
  click() { this.clicked = true; }
  remove() { this.removed = true; }
}

/** 启动独立脚本沙箱，所有网络响应均为测试对象。 */
async function boot(options = {}) {
  const ids = new Map(), calls = [], allCalls = [], downloads = [], revoked = [];
  const element = id => { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); };
  const answerPanel = new Element(), body = new Element();
  const documents = options.documents ?? [{documentId: 'a', chunkCount: 1}];
  const context = {
    document: {getElementById: element, createElement: tag => new Element(tag),
      querySelector: () => answerPanel, body},
    Option: class extends Element {
      /** 构造安全文本选项。 */
      constructor(text, value) { super('option', text); this.value = value; }
    },
    FormData: class { append(key, value) { this[key] = value; } },
    crypto: {randomUUID: () => reportId},
    Blob,
    URL: {createObjectURL: blob => {
      if (options.downloadError) throw new Error('secret-download');
      downloads.push(blob); return 'blob:local-test';
    }, revokeObjectURL: url => revoked.push(url)},
    setTimeout: callback => setImmediate(callback),
    fetch: async (url, request) => {
      allCalls.push({url, request});
      if (url === '/api/reports') {
        if (options.reportsError) throw new Error('secret');
        if (options.reportsDisabled) return {ok: false, json: async () => ({code: 'reports_disabled'})};
        if (request.method === 'POST') return options.reportResponse(JSON.parse(request.body));
        return {ok: true, json: async () => options.reports ? options.reports() : []};
      }
      if (url === '/api/uploads' && request.method !== 'POST') {
        if (options.uploadStateError) throw new Error('secret');
        return {ok: true, json: async () => options.uploads ? options.uploads() : []};
      }
      calls.push({url, request});
      if (url === '/api/uploads') return options.uploadResponse();
      if (url === '/api/documents') {
        if (options.catalogError) throw new Error();
        return {ok: true, json: async () => ({documents})};
      }
      if (options.respond) return options.respond();
      return {ok: true, json: async () => ({result: answer()})};
    }
  };
  vm.runInNewContext(source, context);
  await new Promise(resolve => setImmediate(resolve));
  /** 触发一次表单提交。 */
  const send = () => { element('query').value = 'q'; return element('question-form').listeners.submit({preventDefault() {}}); };
  return {element, calls, allCalls, send, body, answerPanel, downloads, revoked,
    upload: () => element('upload').listeners.click()};
}

/** 构造与Java投影一致的回答，允许逐字段覆盖。 */
function answer(overrides = {}) {
  return {query: 'q', documentId: 'a', status: 'answered', claims: [{text: '<img src=x onerror=alert(1)>',
    citations: [{documentId: 'a', pageNumber: 7, chunkId: 'a7', quote: '<script>unsafe()</script>'}]}],
    missingInfo: [], validation: {semanticSupport: 'not_verified', citationIdentityAndQuote: 'passed',
      statusBasis: 'model_self_assessment'}, usage: {promptTokens: 10, completionTokens: 2, totalTokens: 12}, ...overrides};
}

/** 读取纯文本树，模拟可见内容检查。 */
function text(element) { return [element.textContent, ...element.children.map(text)].join(' '); }

test('目录加载、单次提问、原文纯文本与用量', async () => {
  const app = await boot();
  assert.equal(app.element('submit').disabled, false);
  await app.send();
  assert.equal(app.calls.length, 2);
  assert.deepEqual(JSON.parse(app.calls[1].request.body), {query: 'q', documentId: 'a'});
  assert.match(text(app.element('result')), /<img src=x onerror=alert\(1\)>/);
  assert.match(text(app.element('result')), /<script>unsafe\(\)<\/script>/);
  assert.match(text(app.element('result')), /第 7 页/);
  assert.match(app.element('usage').textContent, /总计 12/);
  assert.match(text(app.element('result')), /语义支持：未验证/);
});

for (const status of ['partial', 'insufficient']) test(`${status}保留缺失项`, async () => {
  const app = await boot({respond: async () => ({ok: true, json: async () => ({result: answer({status,
    claims: status === 'insufficient' ? [] : answer().claims, missingInfo: ['缺少训练时长']})})})});
  await app.send();
  assert.equal(app.element('badge').dataset.status, status);
  assert.match(text(app.element('result')), /缺少训练时长/);
});

for (const documents of [[], [{documentId: 'a', chunkCount: -1}]]) test('空或非法目录禁止提问', async () => {
  const app = await boot({documents});
  assert.equal(app.element('submit').disabled, true);
});

test('目录失败可手动刷新且不自动重试', async () => {
  const app = await boot({catalogError: true});
  assert.equal(app.calls.length, 1);
  assert.equal(app.element('submit').disabled, true);
  assert.equal(app.element('refresh').disabled, false);
});

test('执行时禁用输入且阻止重复发送', async () => {
  let resolve;
  const app = await boot({respond: () => new Promise(r => {resolve = r;})});
  const first = app.send();
  assert.equal(app.element('submit').disabled, true);
  assert.equal(app.element('document').disabled, true);
  assert.equal(app.element('query').disabled, true);
  await app.send();
  assert.equal(app.calls.length, 2);
  resolve({ok: true, json: async () => ({result: answer()})});
  await first;
  assert.equal(app.element('submit').disabled, false);
});

for (const code of ['internal_timeout', 'service_busy', 'citation_invalid', 'untrusted']) test(`${code}固定错误不透传正文`, async () => {
  const app = await boot({respond: async () => ({ok: false, json: async () => ({code, message: 'secret', usage: null})})});
  await app.send();
  assert.equal(app.calls.length, 2);
  assert.equal(app.element('badge').dataset.status, 'error');
  assert.doesNotMatch(app.element('request-state').textContent, /secret/);
  assert.match(app.element('usage').textContent, /未知/);
});

test('网络异常无重试且不保留旧答案', async () => {
  const app = await boot({respond: async () => {throw new Error('secret');}});
  app.element('result').append(new Element('p', 'old answer'));
  await app.send();
  assert.equal(app.calls.length, 2);
  assert.equal(app.element('result').children.length, 0);
  assert.match(app.element('request-state').textContent, /不会自动重试/);
});

test('错论文或非法状态不渲染', async () => {
  const app = await boot({respond: async () => ({ok: true, json: async () => ({result: answer({documentId: 'wrong'})})})});
  await app.send();
  assert.equal(app.element('result').children.length, 0);
  assert.equal(app.element('badge').dataset.status, 'error');
});

test('非法或超出JS安全整数的用量不误报为有效数字', async () => {
  const app = await boot({respond: async () => ({ok: true, json: async () => ({result: answer({usage:
    {promptTokens: Number.MAX_SAFE_INTEGER + 1, completionTokens: 0, totalTokens: Number.MAX_SAFE_INTEGER + 1}})})})});
  await app.send();
  assert.match(app.element('usage').textContent, /未知/);
});

test('回答充分性依据不能丢失或冒充已验证', async () => {
  const app = await boot({respond: async () => ({ok: true, json: async () => ({result: answer({validation:
    {semanticSupport: 'not_verified', citationIdentityAndQuote: 'passed', statusBasis: 'verified'}})})})});
  await app.send();
  assert.equal(app.element('badge').dataset.status, 'error');
});

const uploadId = 'pdf_' + 'a'.repeat(64);
const task = status => ({documentId: uploadId, fileName: '<script>paper.pdf</script>', status,
  chunkCount: status === 'ready' ? 2 : 0, errorCode: 'encrypted_pdf'});

test('未选文件、非PDF及超限不发送上传', async () => {
  const app = await boot();
  await app.upload();
  assert.match(app.element('upload-state').textContent, /请选择/);
  for (const file of [{name: 'x.txt', size: 10}, {name: 'x.pdf', size: 10 * 1024 * 1024 + 1}]) {
    app.element('pdf-file').files = [file];
    await app.upload();
    assert.match(app.element('upload-state').textContent, /10MB/);
  }
  assert.equal(app.allCalls.filter(c => c.request.method === 'POST').length, 0);
});

test('上传只提交一次，完成后刷新目录并选择新论文', async () => {
  let uploaded = false, phase = 0;
  const app = await boot({documents: [{documentId: 'a', chunkCount: 1}, {documentId: uploadId, chunkCount: 2}],
    uploads: () => uploaded ? [task(++phase > 1 ? 'ready' : 'processing')] : [],
    uploadResponse: async () => {uploaded = true; return {ok: true, json: async () => task('queued')};}});
  app.element('pdf-file').files = [{name: 'paper.pdf', size: 10}];
  const promise = app.upload();
  assert.equal(app.element('upload').disabled, true);
  await app.upload();
  await promise;
  assert.equal(app.allCalls.filter(c => c.url === '/api/uploads' && c.request.method === 'POST').length, 1);
  assert.equal(app.allCalls.filter(c => c.url === '/api/documents').length, 2);
  assert.equal(app.element('document').value, uploadId);
  assert.equal(app.element('upload').disabled, false);
  assert.match(app.element('upload-state').textContent, /已就绪/);
});

test('处理失败显示安全分类，不透传上游错误正文', async () => {
  const app = await boot({uploads: () => [task('failed')]});
  assert.match(app.element('upload-state').textContent, /加密 PDF/);
  assert.match(app.element('upload-state').textContent, /相同文件重传仍复用失败/);
  const failed = await boot({uploadResponse: async () => ({ok: false,
    json: async () => ({code: 'unknown-secret', message: 'secret'})})});
  failed.element('pdf-file').files = [{name: 'p.pdf', size: 10}];
  await failed.upload();
  assert.doesNotMatch(failed.element('upload-state').textContent, /secret/);
});

test('页面重载观察旧处理中任务，不重新上传', async () => {
  let reads = 0;
  const app = await boot({uploads: () => [task(++reads < 3 ? 'processing' : 'ready')]});
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(app.allCalls.filter(c => c.request.method === 'POST').length, 0);
  assert.equal(app.element('upload').disabled, false);
  assert.match(app.element('upload-state').textContent, /已就绪/);
});

test('状态网络失败停止轮询，不重新提交任务', async () => {
  const app = await boot({uploadStateError: true});
  assert.equal(app.allCalls.filter(c => c.url === '/api/uploads').length, 1);
  assert.match(app.element('upload-state').textContent, /任务状态不可用/);
});

const reportId = '12345678-1234-4234-8234-123456789abc';
const reportFields = ['research_question', 'datasets', 'model', 'baselines', 'training_settings', 'evaluation_metrics', 'results'];
function reportTask(status = 'ready') {
  return {requestId: reportId, documentId: 'a', experimentScope: 'Alpha on X', status,
    completedFields: status === 'ready' ? 7 : 0, currentField: null, errorCode: status === 'failed' ? 'interrupted' : null,
    generation: status === 'ready' ? {generationCalls: 14, unknownUsageCalls: 1,
      knownUsageSubtotal: {promptTokens: 10, completionTokens: 2, totalTokens: 12}, totalUsage: null} : null,
    result: status === 'ready' ? {sections: reportFields.map(field => ({field, answer: answer({validation:
      {semanticSupport: 'not_verified', citationIdentityAndQuote: 'passed', statusBasis: 'generation_and_review_model_assessment'}})})),
      semanticSupport: 'not_verified', scopeMatch: 'not_verified', crossFieldConsistency: 'not_verified', exhaustiveness: 'not_verified'} : null};
}
function fillReport(app) {
  for (const key of ['model', 'task', 'dataset', 'split']) app.element('report-' + key).value = 'test';
  app.element('report-scope').value = 'Alpha on X';
  app.element('report-model-type').value = 'single';
}

test('重载读取完整报告，纯文本引用、七字段及未知用量小计', async () => {
  const app = await boot({reports: () => [reportTask()]});
  assert.match(text(app.element('report-result')), /研究问题/);
  assert.match(text(app.element('report-result')), /实验结果/);
  assert.match(text(app.element('report-result')), /<script>unsafe\(\)<\/script>/);
  assert.match(text(app.element('report-result')), /已知小计 12 tokens（非完整总量）/);
  assert.match(text(app.element('report-result')), /未验证/);
  assert.equal(app.allCalls.filter(c => c.request.method === 'POST').length, 0);
});

test('报告仅提交一次，重复点击被阻止；完成后读取持久任务', async () => {
  let resolve, posted = false;
  const app = await boot({reports: () => posted ? [reportTask()] : [],
    reportResponse: () => new Promise(r => {resolve = r;})});
  fillReport(app);
  const first = app.element('report-submit').listeners.click();
  await app.element('report-submit').listeners.click();
  assert.equal(app.element('submit').disabled, true);
  assert.equal(app.allCalls.filter(c => c.url === '/api/reports' && c.request.method === 'POST').length, 1);
  posted = true;
  resolve({ok: true, json: async () => reportTask('queued')});
  await first;
  assert.match(app.element('report-state').textContent, /已完成/);
  assert.equal(app.element('submit').disabled, false);
});

test('报告重载只轮询不重新生成，任务完成恢复问答', async () => {
  let reads = 0;
  const app = await boot({reports: () => [reportTask(++reads < 3 ? 'processing' : 'ready')]});
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(app.allCalls.filter(c => c.request.method === 'POST').length, 0);
  assert.equal(app.element('submit').disabled, false);
  assert.match(app.element('report-state').textContent, /7\/7/);
});

test('读取失败停止轮询，提交未知不重试且不反射错误正文', async () => {
  const app = await boot({reportsError: true});
  assert.equal(app.allCalls.filter(c => c.url === '/api/reports').length, 1);
  assert.match(app.element('report-state').textContent, /刷新报告状态/);
  assert.equal(app.element('report-submit').disabled, true);
  const failed = await boot({reportResponse: async () => ({ok: false, json: async () => ({code: 'secret', message: 'secret'})})});
  fillReport(failed);
  await failed.element('report-submit').listeners.click();
  await failed.element('report-submit').listeners.click();
  assert.equal(failed.allCalls.filter(c => c.request.method === 'POST').length, 1);
  assert.doesNotMatch(failed.element('report-state').textContent, /secret/);
  await failed.element('report-refresh').listeners.click();
  assert.equal(failed.element('report-submit').disabled, false);
});

for (const mutate of [t => t.result.sections.reverse(), t => t.result.semanticSupport = 'verified',
  t => t.generation.generationCalls = 15, t => t.result.sections[0].answer.documentId = 'wrong']) {
  test('畸形报告拒绝展示、不自动重提', async () => {
    const task = reportTask(); mutate(task);
    const app = await boot({reports: () => [task]});
    assert.equal(app.element('report-result').children.length, 0);
    assert.equal(app.element('report-submit').disabled, true);
  });
}

test('缺少目标或重复条件不发送报告，失败任务不展示半份结果', async () => {
  const app = await boot({reports: () => [reportTask('failed')]});
  assert.match(app.element('report-state').textContent, /中断/);
  await app.element('report-submit').listeners.click();
  fillReport(app);
  app.element('report-conditions').value = 'decoding, decoding';
  await app.element('report-submit').listeners.click();
  assert.equal(app.allCalls.filter(c => c.request.method === 'POST').length, 0);
});

test('未配置报告目录不影响已有静态论文问答', async () => {
  const app = await boot({reportsDisabled: true});
  assert.equal(app.element('report-submit').disabled, true);
  assert.equal(app.element('submit').disabled, false);
  await app.send();
  assert.equal(app.element('badge').dataset.status, 'answered');
});

test('历史报告可切换，按任务身份而不是当前论文选择展示', async () => {
  const first = reportTask(), second = reportTask();
  second.requestId = '22345678-1234-4234-8234-123456789abc';
  second.experimentScope = 'Second experiment';
  const app = await boot({reports: () => [first, second]});
  app.element('report-tasks').value = first.requestId;
  await app.element('report-tasks').listeners.change();
  assert.match(text(app.element('report-result')), /Alpha on X/);
  assert.doesNotMatch(text(app.element('report-result')), /Second experiment/);
});

test('草稿本地导出包含七字段、精确摘录、身份和用量边界，不新增网络调用', async () => {
  const task = reportTask(); task.privateAudit = 'secret-audit';
  const app = await boot({reports: () => [task]});
  assert.equal(app.element('report-export').disabled, false);
  const before = app.allCalls.length;
  app.element('report-export').listeners.click();
  assert.equal(app.allCalls.length, before);
  assert.equal(app.downloads.length, 1);
  const content = await app.downloads[0].text();
  assert.match(content, /PaperTrace 实验报告草稿/);
  assert.match(content, /已知小计（非完整总量）/);
  assert.match(content, /未知用量调用：1/);
  assert.match(content, /语义支持.*未验证/);
  assert.match(content, /PDF 第 7 页/);
  assert.match(content, /分块：a7/);
  assert.match(content, /<script>unsafe\(\)<\/script>/);
  assert.match(content, /## 7\. 实验结果/);
  assert.doesNotMatch(content, /secret-audit/);
  const link = app.body.children.at(-1);
  assert.equal(link.download, `PaperTrace-草稿-${reportId}.md`);
  assert.equal(link.clicked, true); assert.equal(link.removed, true);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(app.revoked, ['blob:local-test']);
});

test('包含代码围栏的摘录原样保留在更长围栏内，待核对项不丢失', async () => {
  const task = reportTask();
  const a = task.result.sections[0].answer;
  a.status = 'partial'; a.missingInfo = ['需核对 ``` 与 <img>'];
  a.claims[0].citations[0].quote = '```\n<script>unsafe()</script>\n````';
  const app = await boot({reports: () => [task]});
  app.element('report-export').listeners.click();
  const content = await app.downloads[0].text();
  assert.ok(content.includes('`````text\n' + a.claims[0].citations[0].quote + '\n`````'));
  assert.ok(content.includes(a.missingInfo[0]));
});

for (const status of ['queued', 'processing', 'failed']) {
  test(`非ready报告不允许导出：${status}`, async () => {
    // queued/processing的轮询测试用失败读取结束，避免无限模拟轮询。
    let reads = 0;
    const app = await boot({reports: () => {
      if (++reads > 1) throw new Error('stop polling');
      return [reportTask(status)];
    }});
    app.element('report-export').listeners.click();
    assert.equal(app.downloads.length, 0);
    assert.equal(app.element('report-export').disabled, true);
  });
}

test('切换历史报告按所选任务导出，读取失败清除旧导出状态', async () => {
  const first = reportTask(), second = reportTask();
  second.requestId = '22345678-1234-4234-8234-123456789abc'; second.experimentScope = 'Second experiment';
  let fail = false;
  const app = await boot({reports: () => {if (fail) throw new Error('secret'); return [first, second];}});
  app.element('report-tasks').value = first.requestId;
  await app.element('report-tasks').listeners.change();
  app.element('report-export').listeners.click();
  assert.doesNotMatch(await app.downloads[0].text(), /Second experiment/);
  fail = true;
  await app.element('report-refresh').listeners.click();
  app.element('report-export').listeners.click();
  assert.equal(app.downloads.length, 1);
  assert.equal(app.element('report-export').disabled, true);
});

test('下载失败显示安全提示，报告不重新生成', async () => {
  const app = await boot({reports: () => [reportTask()], downloadError: true});
  const before = app.allCalls.length;
  app.element('report-export').listeners.click();
  assert.equal(app.allCalls.length, before);
  assert.match(app.element('report-export-state').textContent, /导出失败/);
  assert.doesNotMatch(app.element('report-export-state').textContent, /secret/);
});
