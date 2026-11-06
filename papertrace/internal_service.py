"""提供仅供本机业务后端调用的 PaperTrace 问答 HTTP 服务。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from papertrace.qa import QAError, answer_question


SERVICE_VERSION = "papertrace-internal-question-v1"
MAX_REQUEST_BYTES = 256 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ServiceError(ValueError):
    """表示内部服务配置或请求不符合安全边界。"""


class QuestionService:
    """将 HTTP 请求映射到既有的单问题证据问答流程。"""

    def __init__(self, chunks, retriever_factory: Callable, generator, *, top_k: int = 5,
                 max_tokens: int = 2048) -> None:
        """显式注入语料、检索器工厂和生成器，不读取密钥或创建隐式客户端。"""
        if not isinstance(chunks, list):
            raise ServiceError("chunks 必须是列表")
        if not callable(retriever_factory) or not callable(getattr(generator, "generate", None)):
            raise ServiceError("必须提供检索器工厂和生成器")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ServiceError("top_k 必须是1到20的整数")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ServiceError("max_tokens 必须是1到8192的整数")
        self.chunks = chunks
        self.retriever_factory = retriever_factory
        self.generator = generator
        self.top_k = top_k
        self.max_tokens = max_tokens

    def answer(self, query: str, document_id: str) -> dict:
        """执行一条论文范围内问答并返回既有安全结果结构。"""
        if not isinstance(query, str) or not query.strip():
            raise ServiceError("query 必须是非空字符串")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ServiceError("document_id 必须是非空字符串")
        return answer_question(
            query, document_id, self.chunks,
            retriever_factory=self.retriever_factory, generator=self.generator,
            top_k=self.top_k, max_tokens=self.max_tokens,
        )


def _json_bytes(value: dict) -> bytes:
    """序列化响应并拒绝非标准 JSON 数值。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _error(code: str, message: str, status: int) -> tuple[int, bytes]:
    """构造不回显问题、候选或模型正文的固定错误响应。"""
    body = {"ok": False, "error": {"code": code, "message": message, "service_version": SERVICE_VERSION}}
    return status, _json_bytes(body)


def make_handler(service: QuestionService):
    """为显式注入的服务实例创建无访问日志 HTTP handler。"""
    if not isinstance(service, QuestionService):
        raise ServiceError("service 必须是 QuestionService")

    class Handler(BaseHTTPRequestHandler):
        """处理健康检查和单问题问答请求。"""

        server_version = "PaperTraceInternal/1"
        sys_version = ""

        def log_message(self, format, *args):
            """禁用默认访问日志，避免问题正文进入标准错误。"""
            return

        def _send(self, status: int, body: bytes) -> None:
            """发送 JSON 响应并关闭连接。"""
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """处理健康检查，其余路径返回固定 404。"""
            if self.path == "/healthz":
                self._send(200, _json_bytes({"ok": True, "service_version": SERVICE_VERSION}))
            else:
                self._send(*_error("not_found", "路径不存在", 404))

        def do_POST(self):
            """解析单问题 JSON 请求并调用注入的问答服务。"""
            if self.path != "/v1/question":
                self._send(*_error("not_found", "路径不存在", 404))
                return
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError:
                length = -1
            if length < 0:
                self._send(*_error("invalid_content_length", "请求长度无效", 400))
                return
            if length > MAX_REQUEST_BYTES:
                # 读取有限前缀再回复，避免客户端仍在发送请求体时连接被系统重置。
                self.rfile.read(MAX_REQUEST_BYTES + 1)
                self._send(*_error("request_too_large", "请求体超过大小上限", 413))
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError
                payload = json.loads(raw.decode("utf-8"))
                if (not isinstance(payload, dict) or set(payload) != {"query", "document_id"}
                        or not isinstance(payload["query"], str)
                        or not isinstance(payload["document_id"], str)):
                    raise ValueError
            except (ValueError, UnicodeError, json.JSONDecodeError, RecursionError):
                self._send(*_error("invalid_json", "请求必须是包含 query 和 document_id 的 JSON 对象", 400))
                return
            try:
                result = service.answer(payload["query"], payload["document_id"])
                body = _json_bytes({"ok": True, "service_version": SERVICE_VERSION, "result": result})
            except QAError as error:
                self._send(*_error(error.code, "问答请求未完成", 422))
                return
            except ServiceError:
                self._send(*_error("invalid_input", "请求字段无效", 400))
                return
            except Exception:
                self._send(*_error("internal_error", "内部服务执行失败", 500))
                return
            self._send(200, body)

    return Handler


def serve(service: QuestionService, host: str = "127.0.0.1", port: int = 8765):
    """在本机地址启动线程 HTTP 服务，禁止公网监听。"""
    if not isinstance(host, str) or host.casefold() not in _LOCAL_HOSTS:
        raise ServiceError("内部服务仅允许监听本机地址")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ServiceError("port 必须是0到65535的整数")
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server
