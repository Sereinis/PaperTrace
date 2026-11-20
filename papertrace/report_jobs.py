"""持久化本机报告任务，串行复用报告引擎，不自动重试或恢复计费任务。"""

import copy
import json
import re
import threading
from pathlib import Path

from papertrace.embedding_cache import atomic_write_json
from papertrace.experiment_report import build_experiment_report, ReportError, _FIELD_BY_ID
from papertrace.generation import safe_usage
from papertrace.internal_service import ServiceError, _strict_object, _reject_constant
from papertrace.result_graph import validate_target

FIELDS = tuple(_FIELD_BY_ID)
_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")


def _text(value, limit):
    """检查有界UTF-8文本，不读取密钥或回显非法输入。"""
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ServiceError()
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ServiceError() from None
    return value


def _request(payload):
    """固定七字段与显式目标，不允许客户端选择路径、生成器或任意参数。"""
    if not isinstance(payload, dict) or set(payload) != {"request_id", "document_id", "experiment_scope", "result_target"}:
        raise ServiceError()
    if not isinstance(payload["request_id"], str) or not _ID.fullmatch(payload["request_id"]):
        raise ServiceError()
    _text(payload["document_id"], 256)
    _text(payload["experiment_scope"], 1000)
    try:
        validate_target(payload["result_target"])
    except (ValueError, TypeError):
        raise ServiceError() from None
    return copy.deepcopy(payload)


def _generation(value):
    """仅投影合法调用次数与已知用量，未知保持为空。"""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("报告用量无效")
    calls, unknown = value.get("generation_calls"), value.get("unknown_usage_calls")
    subtotal, total = safe_usage(value.get("known_usage_subtotal")), safe_usage(value.get("total_usage"))
    if (type(calls) is not int or not 0 <= calls <= 14 or type(unknown) is not int
            or not 0 <= unknown <= calls or subtotal is None
            or (calls > 0 and unknown == 0 and total != subtotal)
            or ((unknown > 0 or calls == 0) and value.get("total_usage") is not None)):
        raise ValueError("报告用量无效")
    return {"generation_calls": calls, "unknown_usage_calls": unknown,
            "known_usage_subtotal": subtotal, "total_usage": total}


def project_report(report):
    """只输出结论、精确引用和未验证标识，不发布候选全文或复核审计。"""
    sections = []
    _text(report["document_id"], 256)
    _text(report["experiment_scope"], 1000)
    if not isinstance(report["sections"], list) or [s["field"] for s in report["sections"]] != list(FIELDS):
        raise ValueError("报告字段无效")
    for section in report["sections"]:
        qa = section["qa"]
        answer = {key: copy.deepcopy(qa[key]) for key in
                  ("query", "document_id", "status", "claims", "missing_info", "validation")}
        usage = (qa.get("generation") or {}).get("usage")
        answer["generation"] = {"usage": safe_usage(usage)}
        _validate_answer(answer, report["document_id"])
        sections.append({"field": section["field"], "label": section["label"], "qa": answer})
    validation = {key: report["validation"][key] for key in
                  ("semantic_support", "experiment_scope_match", "cross_field_consistency", "exhaustiveness")}
    if any(value != "not_verified" for value in validation.values()):
        raise ValueError("报告校验状态无效")
    return {"document_id": report["document_id"], "experiment_scope": report["experiment_scope"],
            "sections": sections, "generation": _generation(report["generation"]),
            "validation": validation}


def _validate_answer(answer, document):
    """校验公开结论和引用结构；定位正确仍不等于语义支持。"""
    _text(answer["query"], 4000)
    status, claims, missing = answer["status"], answer["claims"], answer["missing_info"]
    if (answer["document_id"] != document or status not in {"answered", "partial", "insufficient"}
            or not isinstance(claims, list) or not isinstance(missing, list)):
        raise ValueError("报告答案无效")
    for item in missing:
        _text(item, 8000)
    if ((status == "answered" and (not claims or missing))
            or (status == "partial" and (not claims or not missing))
            or (status == "insufficient" and (claims or not missing))):
        raise ValueError("报告答案状态不一致")
    validation = answer["validation"]
    if (validation.get("semantic_support") != "not_verified"
            or validation.get("citation_identity_and_quote") not in {"passed", "not_applicable"}
            or validation.get("status_basis") not in {"model_self_assessment", "no_retrieved_evidence",
                "generation_and_review_model_assessment", "structured_identity_link_checks_and_model_review"}):
        raise ValueError("报告答案校验状态无效")
    answer["validation"] = {key: validation[key] for key in
                            ("semantic_support", "citation_identity_and_quote", "status_basis")}
    clean_claims = []
    for claim in claims:
        _text(claim["text"], 16000)
        if not isinstance(claim["citations"], list) or not claim["citations"]:
            raise ValueError("报告缺少引用")
        refs = []
        for citation in claim["citations"]:
            for key in ("evidence_id", "segment_id", "chunk_id", "quote"):
                _text(citation[key], 64000)
            start, end, page = (citation[k] for k in ("quote_start", "quote_end", "page_number"))
            if (citation["document_id"] != document or any(type(n) is not int for n in (start, end, page))
                    or page < 1 or start < 0 or end - start != len(citation["quote"])):
                raise ValueError("报告引用身份或位置无效")
            refs.append({key: citation[key] for key in ("evidence_id", "segment_id", "chunk_id", "document_id",
                         "page_number", "quote", "quote_start", "quote_end")})
        clean_claims.append({"text": claim["text"], "citations": refs})
    answer["claims"] = clean_claims


class ReportJobs:
    """同一服务仅一个问答/报告执行；最多保留50个本地任务。"""

    def __init__(self, service, root, *, builder=build_experiment_report):
        """恢复已保存任务，未完成任务标为中断，不自动继续或重试。"""
        self.service, self.root, self.builder = service, Path(root).resolve(), builder
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard, self._tasks = threading.Lock(), {}
        self._thread = None
        paths = list(self.root.glob("*.json"))
        if len(paths) > 50:
            raise ValueError("报告任务数量超限")
        for path in paths:
            if path.is_symlink() or not _ID.fullmatch(path.stem) or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("报告任务存储无效")
            task = json.loads(path.read_bytes(), object_pairs_hook=_strict_object, parse_constant=_reject_constant)
            self._validate_stored(task, path.stem)
            if task["status"] in {"queued", "processing"}:
                task.update(status="failed", current_field=None, error_code="interrupted", result=None)
                atomic_write_json(task, path)
            self._tasks[path.stem] = task

    def _validate_stored(self, task, identity):
        """校验持久化任务身份和状态，损坏时拒绝恢复。"""
        if not isinstance(task, dict) or set(task) != {"request", "status", "completed_fields", "current_field",
                                                      "error_code", "generation", "result"}:
            raise ValueError("报告任务无效")
        payload = _request(task["request"])
        if (payload["request_id"] != identity or task["status"] not in {"queued", "processing", "ready", "failed"}
                or type(task["completed_fields"]) is not int or not 0 <= task["completed_fields"] <= 7
                or task["current_field"] not in (*FIELDS, None)
                or task["error_code"] not in (None, "interrupted", "field_failed", "report_failed")):
            raise ValueError("报告任务状态无效")
        if ((task["status"] == "failed") != (task["error_code"] is not None)
                or (task["status"] in {"ready", "failed", "queued"} and task["current_field"] is not None)
                or (task["status"] == "queued" and task["completed_fields"] != 0)):
            raise ValueError("报告任务状态不一致")
        _generation(task["generation"])
        if task["status"] == "ready":
            result = task["result"]
            if (task["completed_fields"] != 7 or not isinstance(result, dict)
                    or result.get("document_id") != payload["document_id"]
                    or result.get("experiment_scope") != payload["experiment_scope"]
                    or [s.get("field") for s in result.get("sections", [])] != list(FIELDS)):
                raise ValueError("报告结果无效")
            if result.get("generation") != task["generation"] or task["generation"] is None:
                raise ValueError("报告用量不一致")
            # 只允许已知公开字段，避免任务文件混入完整候选或内部审计。
            if project_report(result) != result:
                raise ValueError("报告公开字段无效")
        elif task["result"] is not None:
            raise ValueError("未完成任务不能发布报告")

    def _public(self, task):
        """返回脱离内部状态的任务副本，不返回路径、密钥或候选。"""
        return copy.deepcopy(task)

    def list(self):
        """读取任务快照，不触发模型或重跑已失败任务。"""
        with self._guard:
            return [self._public(task) for task in self._tasks.values()]

    def submit(self, payload):
        """先持久化幂等请求再创建后台任务，同一请求身份不会重复计费。"""
        payload = _request(payload)
        identity = payload["request_id"]
        with self._guard:
            if identity in self._tasks:
                task = self._tasks[identity]
                if task["request"] != payload:
                    raise ServiceError("request_conflict")
                return self._public(task)
            if len(self._tasks) >= 50:
                raise ServiceError("report_limit")
            imported = self.service.library.service(payload["document_id"]) if self.service.library else None
            selected = imported if imported is not None else self.service
            if payload["document_id"] not in selected.document_ids:
                raise ServiceError("document_not_found")
            if not self.service._lock.acquire(blocking=False):
                raise ServiceError("service_busy")
            try:
                task = {"request": payload, "status": "queued", "completed_fields": 0, "current_field": None,
                        "error_code": None, "generation": None, "result": None}
                atomic_write_json(task, self.root / f"{identity}.json")
                self._tasks[identity] = task
                self._thread = threading.Thread(target=self._run, args=(identity, selected), daemon=True)
                self._thread.start()
            except Exception:
                if identity in self._tasks:
                    task.update(status="failed", current_field=None, error_code="report_failed", result=None)
                    try:
                        atomic_write_json(task, self.root / f"{identity}.json")
                    except Exception:
                        pass
                self.service._lock.release()
                raise
            return self._public(task)

    def _update(self, identity, **values):
        """在锁内先原子写盘再替换可见状态，读取不会观察半份更新。"""
        with self._guard:
            task = copy.deepcopy(self._tasks[identity])
            task.update(values)
            atomic_write_json(task, self.root / f"{identity}.json")
            self._tasks[identity] = task

    def _run(self, identity, selected):
        """逐字段生成并记录进度，失败只发布安全分类和合法用量。"""
        try:
            payload = self._tasks[identity]["request"]

            def progress(field, completed):
                """只发布固定字段与完成计数，不提前展示未完成报告。"""
                self._update(identity, status="processing", current_field=field, completed_fields=completed)

            report = self.builder(payload["document_id"], payload["experiment_scope"], selected.chunks,
                                  retriever_factory=selected.retriever_factory, generator=selected.generator,
                                  top_k=selected.top_k, max_tokens=selected.max_tokens,
                                  result_target=payload["result_target"], progress=progress)
            result = project_report(report)
            if (result["document_id"] != payload["document_id"]
                    or result["experiment_scope"] != payload["experiment_scope"]
                    or result["generation"] is None):
                raise ValueError("报告请求身份不一致")
            self._update(identity, status="ready", completed_fields=7, current_field=None,
                         result=result, generation=result["generation"])
        except Exception as error:
            generation = None
            if isinstance(error, ReportError):
                try:
                    generation = _generation(error.diagnostics.get("generation"))
                except (ValueError, TypeError):
                    pass
            try:
                self._update(identity, status="failed", current_field=None,
                             error_code="field_failed" if isinstance(error, ReportError) else "report_failed",
                             generation=generation, result=None)
            except Exception:
                # 存储不可用时仅在内存标记失败；重启仍会把磁盘未完成状态标为中断。
                with self._guard:
                    self._tasks[identity].update(status="failed", current_field=None,
                                                  error_code="report_failed", result=None)
        finally:
            self.service._lock.release()
