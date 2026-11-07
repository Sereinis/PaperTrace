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
}

/** 启动独立脚本沙箱，所有网络响应均为测试对象。 */
async function boot(options = {}) {
  const ids = new Map(), calls = [], allCalls = [];
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
    setTimeout: callback => setImmediate(callback),
    fetch: async (url, request) => {
      allCalls.push({url, request});
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
  return {element, calls, allCalls, send, body, answerPanel,
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
