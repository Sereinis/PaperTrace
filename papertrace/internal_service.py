"""提供仅供本机业务后端调用的 PaperTrace 问答 HTTP 服务。"""

from __future__ import annotations

import json
import copy
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from papertrace.qa import QAError, answer_question
from papertrace.embedding_cache import validate_chunks
from papertrace.generation import safe_usage


SERVICE_VERSION = "papertrace-internal-question-v1"
MAX_REQUEST_BYTES = 256 * 1024
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ServiceError(ValueError):
    """表示内部服务配置或请求不符合安全边界。"""

    def __init__(self, code: str = "invalid_input") -> None:
        """只保存固定分类，不携带问题或配置正文。"""
        super().__init__(code)
        self.code = code


class QuestionService:
    """将 HTTP 请求映射到既有的单问题证据问答流程。"""

    def __init__(self, chunks, retriever_factory: Callable, generator, *, top_k: int = 5,
                 max_tokens: int = 2048) -> None:
        """显式注入语料、检索器工厂和生成器，不读取密钥或创建隐式客户端。"""
        if not isinstance(chunks, list):
            raise ServiceError("chunks 必须是列表")
        validate_chunks(chunks)
        if not callable(retriever_factory) or not callable(getattr(generator, "generate", None)):
            raise ServiceError("必须提供检索器工厂和生成器")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ServiceError("top_k 必须是1到20的整数")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ServiceError("max_tokens 必须是1到8192的整数")
        self.chunks = copy.deepcopy(chunks)
        self.document_ids = frozenset(chunk["document_id"] for chunk in chunks)
        self._lock = threading.Lock()
        self.retriever_factory = retriever_factory
        self.generator = generator
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.library = None

    def documents(self) -> list[dict]:
        """只列出已加载论文身份与分块数量，不返回原文、路径或模型配置。"""
        counts = {}
        for chunk in self.chunks:
            identity = chunk["document_id"]
            counts[identity] = counts.get(identity, 0) + 1
        if self.library is not None:
            for row in self.library.documents():
                counts[row["document_id"]] = row["chunk_count"]
        return [{"document_id": identity, "chunk_count": counts[identity]} for identity in sorted(counts)]

    def answer(self, query: str, document_id: str) -> dict:
        """执行一条论文范围内问答并返回既有安全结果结构。"""
        if not isinstance(query, str) or not query.strip():
            raise ServiceError("query 必须是非空字符串")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ServiceError("document_id 必须是非空字符串")
        imported = self.library.service(document_id) if self.library is not None else None
        if document_id not in self.document_ids and imported is None:
            raise ServiceError("document_not_found")
        if not self._lock.acquire(blocking=False):
            raise ServiceError("service_busy")
        try:
            if imported is not None:
                return imported.answer(query, document_id)
            return answer_question(
                query, document_id, self.chunks,
                retriever_factory=self.retriever_factory, generator=self.generator,
                top_k=self.top_k, max_tokens=self.max_tokens,
            )
        finally:
            self._lock.release()


def _json_bytes(value: dict) -> bytes:
    """序列化响应并拒绝非标准 JSON 数值。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _error(code: str, message: str, status: int, usage=None) -> tuple[int, bytes]:
    """构造不回显问题、候选或模型正文的固定错误响应。"""
    body = {"ok": False, "error": {"code": code, "message": message, "service_version": SERVICE_VERSION}}
    body["error"]["usage"] = safe_usage(usage)
    return status, _json_bytes(body)


def _strict_object(pairs):
    """拒绝重复JSON字段，避免不同组件对请求理解不一致。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("重复字段")
        result[key] = value
    return result


def _reject_constant(value):
    """拒绝JSON之外的非有限数值。"""
    raise ValueError("非法常量")


def make_handler(service: QuestionService):
    """为显式注入的服务实例创建无访问日志 HTTP handler。"""
    if not isinstance(service, QuestionService):
        raise ServiceError("service 必须是 QuestionService")

    class Handler(BaseHTTPRequestHandler):
        """处理健康检查和单问题问答请求。"""

        server_version = "PaperTraceInternal/1"
        sys_version = ""

        def setup(self):
            """限制读取请求头和请求体的等待时间，不限制模型执行。"""
            super().setup()
            self.connection.settimeout(5)

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
            elif self.path == "/v1/documents":
                self._send(200, _json_bytes({"ok": True, "service_version": SERVICE_VERSION,
                                            "documents": service.documents()}))
            else:
                self._send(*_error("not_found", "路径不存在", 404))

        def do_POST(self):
            """解析问答或导入JSON请求，拒绝客户端路径及额外字段。"""
            importing = self.path == "/v1/import"
            if self.path not in {"/v1/question", "/v1/import"}:
                self._send(*_error("not_found", "路径不存在", 404))
                return
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                self._send(*_error("invalid_input", "仅接受定长JSON请求", 400))
                return
            if len(self.headers.get_all("Content-Length", [])) != 1:
                self._send(*_error("invalid_input", "请求长度字段无效", 400))
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
                try:
                    self.rfile.read(MAX_REQUEST_BYTES + 1)
                except OSError:
                    pass
                self._send(*_error("request_too_large", "请求体超过大小上限", 413))
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError
                payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                                     parse_constant=_reject_constant)
                required = {"document_id"} if importing else {"query", "document_id"}
                if (not isinstance(payload, dict) or set(payload) != required
                        or (not importing and not isinstance(payload["query"], str))
                        or not isinstance(payload["document_id"], str)):
                    raise ValueError
            except (ValueError, UnicodeError, json.JSONDecodeError, RecursionError):
                self._send(*_error("invalid_json", "请求字段或JSON格式无效", 400))
                return
            except OSError:
                self._send(*_error("request_timeout", "读取请求超时或中断", 408))
                return
            try:
                if importing:
                    from papertrace.library import ImportFailure
                    if service.library is None:
                        self._send(*_error("uploads_disabled", "未配置本地论文库", 503))
                        return
                    try:
                        imported = service.library.import_document(payload["document_id"])
                    except ImportFailure as error:
                        self._send(*_error(error.code, "论文处理未完成，不会自动重试", 422))
                        return
                    self._send(200, _json_bytes({"ok": True, "service_version": SERVICE_VERSION,
                                                "document": imported}))
                    return
                result = service.answer(payload["query"], payload["document_id"])
                body = _json_bytes({"ok": True, "service_version": SERVICE_VERSION, "result": result})
            except QAError as error:
                status = 400 if error.code == "invalid_input" else 422 if error.code == "context_too_large" else 502
                self._send(*_error(error.code, "问答请求未完成；不会自动重试", status,
                                   usage=error.diagnostics.get("usage")))
                return
            except ServiceError as error:
                code = error.code if error.code in {"document_not_found", "service_busy"} else "invalid_input"
                status = {"document_not_found": 404, "service_busy": 503}.get(code, 400)
                self._send(*_error(code, "论文不存在或服务暂不可处理此请求", status))
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
    server_class = ThreadingHTTPServer
    if host == "::1":
        class IPv6Server(ThreadingHTTPServer):
            """只为显式IPv6回环地址使用IPv6套接字。"""
            address_family = socket.AF_INET6

        server_class = IPv6Server
    server = server_class((host, port), make_handler(service))
    server.daemon_threads = True
    return server
