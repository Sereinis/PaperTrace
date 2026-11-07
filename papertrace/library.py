"""本地单用户论文库：固定目录导入、原子产物与不联网的重启恢复。"""

import hashlib
import json
import re
import threading
from pathlib import Path

from pypdf import PdfReader

from papertrace.chunker import chunk_document
from papertrace.embedding_cache import atomic_write_json, embed_chunks
from papertrace.embeddings import SiliconFlowEmbedder
from papertrace.parser import extract_pdf_pages

IDENTITY = re.compile(r"pdf_[0-9a-f]{64}\Z")
MAX_PDF_BYTES = 10 * 1024 * 1024


class ImportFailure(ValueError):
    """只保留安全导入分类，不携带文件内容和本地路径。"""

    def __init__(self, code):
        """保存预定义错误码。"""
        super().__init__(code)
        self.code = code


class PaperLibrary:
    """只读取服务端配置目录，不接受客户端路径；同一进程仅一个导入执行。"""

    def __init__(self, root, *, model="deepseek-flash", top_k=5, max_tokens=2048,
                 generation_timeout=60, embedding_timeout=20, embedder_factory=None,
                 offline_query_cache=False):
        """保存固定配置并从完整就绪清单恢复，不重新解析或向量化。"""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.options = dict(model=model, top_k=top_k, max_tokens=max_tokens,
                            generation_timeout=generation_timeout, embedding_timeout=embedding_timeout)
        self.factory = embedder_factory or (lambda: SiliconFlowEmbedder(
            batch_size=8, timeout=embedding_timeout, max_retries=0))
        self.offline_query_cache = offline_query_cache
        self._import_lock = threading.Lock()
        self._guard = threading.Lock()
        self._services = {}
        for marker in self.root.glob("pdf_*/ready.json"):
            self._restore(marker.parent.name)

    def _folder(self, identity):
        """强制内容哈希身份和固定文件名，拒绝目录链接或路径穿越。"""
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            raise ImportFailure("invalid_upload")
        folder = self.root / identity
        if folder.is_symlink() or folder.resolve().parent != self.root:
            raise ImportFailure("invalid_upload")
        return folder

    def _restore(self, identity):
        """恢复前核对PDF及产物哈希，损坏时拒绝启动，不自动修复或计费。"""
        from papertrace.service_cli import create_service

        folder = self._folder(identity)
        if (folder / "ready.json").is_symlink():
            raise ImportFailure("library_corrupt")
        marker = json.loads((folder / "ready.json").read_bytes())
        if marker.get("schema_version") != 1 or marker.get("document_id") != identity:
            raise ImportFailure("library_corrupt")
        for name in ("source.pdf", "chunks.json", "vectors.json"):
            path = folder / name
            if path.is_symlink() or path.resolve().parent != folder or hashlib.sha256(
                    path.read_bytes()).hexdigest() != marker.get("hashes", {}).get(name):
                raise ImportFailure("library_corrupt")
        if marker["hashes"]["source.pdf"] != identity[4:]:
            raise ImportFailure("library_corrupt")
        service = create_service([folder / "chunks.json"], vectors=folder / "vectors.json",
                                 query_cache=self.root / "query-cache",
                                 offline_query_cache=self.offline_query_cache, **self.options)
        if service.document_ids != {identity}:
            raise ImportFailure("library_corrupt")
        with self._guard:
            self._services[identity] = service
        return len(service.chunks)

    def documents(self):
        """返回可问答文档的新目录副本，不持锁执行模型请求。"""
        with self._guard:
            return [{"document_id": key, "chunk_count": len(service.chunks)}
                    for key, service in sorted(self._services.items())]

    def service(self, identity):
        """取得已完整发布的论文服务，未完成产物不可见。"""
        with self._guard:
            return self._services.get(identity)

    def import_document(self, identity):
        """有限处理一次PDF；重传完整产物复用，失败不自行重跑。"""
        folder = self._folder(identity)
        existing = self.service(identity)
        if existing is not None:
            return {"document_id": identity, "chunk_count": len(existing.chunks)}
        if not self._import_lock.acquire(blocking=False):
            raise ImportFailure("import_busy")
        try:
            existing = self.service(identity)
            if existing is not None:
                return {"document_id": identity, "chunk_count": len(existing.chunks)}
            if not (folder / "source.pdf").is_file():
                raise ImportFailure("invalid_upload")
            # 接口重复调用不续跑中断或失败的计费任务。
            if (folder / "processing.json").exists():
                raise ImportFailure("import_already_attempted")
            audit = {"schema_version": 1, "stage": "validating", "embedding_batches_attempted": 0,
                     "embedding_texts_attempted": 0, "automatic_retries": 0, "usage": None}
            atomic_write_json(audit, folder / "processing.json")
            pdf = folder / "source.pdf"
            if pdf.is_symlink() or pdf.resolve().parent != folder or not 5 <= pdf.stat().st_size <= MAX_PDF_BYTES:
                raise ImportFailure("invalid_upload")
            raw = pdf.read_bytes()
            if not raw.startswith(b"%PDF-") or hashlib.sha256(raw).hexdigest() != identity[4:]:
                raise ImportFailure("invalid_upload")
            reader = PdfReader(pdf)
            if reader.is_encrypted:
                raise ImportFailure("encrypted_pdf")
            if not 1 <= len(reader.pages) <= 60:
                raise ImportFailure("page_limit")
            audit["stage"] = "parsing"
            atomic_write_json(audit, folder / "processing.json")
            parsed = extract_pdf_pages(pdf)
            if parsed["error"] is not None:
                raise ImportFailure("pdf_parse_failed")
            if sum(len(p["cleaned_text"]) for p in parsed["pages"]) > 300_000:
                raise ImportFailure("text_limit")
            chunks = chunk_document(parsed["pages"], identity, max_chars=1200, overlap_chars=100)
            if not chunks:
                raise ImportFailure("no_extractable_text")
            if len(chunks) > 300:
                raise ImportFailure("chunk_limit")
            atomic_write_json({"chunks": chunks, "error": None}, folder / "chunks.json")
            audit.update(stage="embedding", chunk_count=len(chunks))
            atomic_write_json(audit, folder / "processing.json")
            outer = self

            class CountedEmbedder:
                """每个外层批次发送前持久记录尝试数，用量未知不估算。"""

                def __init__(self):
                    """仅缺失向量时才通过既有适配器加载密钥。"""
                    self.client = outer.factory()
                    self.model = self.client.model

                def embed(self, texts):
                    """每批最多8条，不自动重试，失败后保留成功缓存。"""
                    audit["embedding_batches_attempted"] += 1
                    audit["embedding_texts_attempted"] += len(texts)
                    atomic_write_json(audit, folder / "processing.json")
                    return self.client.embed(texts)

            vectors = embed_chunks(chunks, self.root / "embedding-cache", batch_size=8,
                                   embedder_factory=CountedEmbedder)
            atomic_write_json(vectors, folder / "vectors.json")
            marker = {"schema_version": 1, "document_id": identity, "hashes": {
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ("source.pdf", "chunks.json", "vectors.json")}}
            audit.update(stage="ready", cache_stats=vectors["cache_stats"])
            atomic_write_json(audit, folder / "processing.json")
            atomic_write_json(marker, folder / "ready.json")
            count = self._restore(identity)
            return {"document_id": identity, "chunk_count": count}
        except ImportFailure:
            raise
        except Exception:
            raise ImportFailure("import_failed") from None
        finally:
            self._import_lock.release()
