"""验证本地论文库的导入边界、原子产物与重启恢复。"""

import hashlib
import json
from pathlib import Path

import pytest

from papertrace import library
from papertrace.library import ImportFailure, PaperLibrary


@pytest.fixture(autouse=True)
def no_real_clients(monkeypatch):
    """所有测试禁止读取真实模型配置或访问计费接口。"""
    def blocked(*args, **kwargs):
        """真实客户端一旦被构造立即使测试失败。"""
        pytest.fail("unexpected real client")
    monkeypatch.setattr(library, "SiliconFlowEmbedder", blocked)
    monkeypatch.setattr("papertrace.service_cli.SiliconFlowEmbedder", blocked)
    monkeypatch.setattr("papertrace.service_cli.DeepSeekGenerator", blocked)


def _pdf(tmp_path, content=b"%PDF-1.7\nfixture"):
    """以内容哈希建立测试目录和源文件。"""
    raw = tmp_path / "source.pdf"
    raw.write_bytes(content)
    identity = "pdf_" + hashlib.sha256(content).hexdigest()
    folder = tmp_path / identity
    folder.mkdir()
    (folder / "source.pdf").write_bytes(content)
    return identity, folder


def test_identity_and_invalid_upload_are_fixed(tmp_path):
    """客户端身份不允许路径或大写哈希。"""
    lib = PaperLibrary(tmp_path)
    with pytest.raises(ImportFailure, match="invalid_upload"):
        lib.import_document("../secret")
    with pytest.raises(ImportFailure, match="invalid_upload"):
        lib.import_document("pdf_" + "A" * 64)
    assert library.IDENTITY.fullmatch("pdf_" + "a" * 64)


def test_import_rejects_bad_signature_and_hash(tmp_path):
    """签名和内容哈希不符时不进入解析。"""
    identity, folder = _pdf(tmp_path, b"not a pdf")
    lib = PaperLibrary(tmp_path)
    with pytest.raises(ImportFailure, match="invalid_upload"):
        lib.import_document(identity)
    folder.joinpath("processing.json").unlink()
    folder.joinpath("source.pdf").write_bytes(b"%PDF-1.7\nchanged")
    with pytest.raises(ImportFailure, match="invalid_upload"):
        lib.import_document(identity)


def test_import_publishes_atomic_ready_and_restores(tmp_path, monkeypatch):
    """仅完整产物发布后才出现在论文目录。"""
    identity, folder = _pdf(tmp_path)
    chunks = [{"chunk_id": "p_c", "document_id": identity, "page_number": 1, "text": "attention"}]

    class FakeReader:
        is_encrypted = False
        pages = [object()]

    class FakeService:
        def __init__(self):
            """保存模拟论文身份和分块。"""
            self.chunks = chunks
            self.document_ids = {identity}

    vector = {"schema_version": 1, "embedding_config": {"model": "test"},
              "chunk_count": 1, "chunks_sha256": "test", "records": [],
              "cache_stats": {"cache_hit_texts": 0, "embedded_texts": 1}}
    monkeypatch.setattr(library, "PdfReader", lambda path: FakeReader())
    monkeypatch.setattr(library, "extract_pdf_pages", lambda path: {
        "error": None, "pages": [{"page_number": 1, "cleaned_text": "attention"}]})
    monkeypatch.setattr(library, "chunk_document", lambda *args, **kwargs: chunks)
    monkeypatch.setattr(library, "embed_chunks", lambda *args, **kwargs: vector)
    monkeypatch.setattr("papertrace.service_cli.create_service", lambda *args, **kwargs: FakeService())
    lib = PaperLibrary(tmp_path, embedder_factory=lambda: pytest.fail("must not be used by fake embed"))
    result = lib.import_document(identity)
    assert result == {"document_id": identity, "chunk_count": 1}
    assert lib.documents() == [{"document_id": identity, "chunk_count": 1}]
    assert all(folder.joinpath(name).exists() for name in ("source.pdf", "chunks.json", "vectors.json", "ready.json"))
    restored = PaperLibrary(tmp_path)
    assert restored.documents() == [{"document_id": identity, "chunk_count": 1}]


def test_ready_hash_corruption_is_rejected(tmp_path, monkeypatch):
    """损坏产物在创建服务前拒绝恢复。"""
    identity, folder = _pdf(tmp_path)
    for name, value in (("chunks.json", {"chunks": []}), ("vectors.json", {"records": []})):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
              for name in ("source.pdf", "chunks.json", "vectors.json")}
    (folder / "ready.json").write_text(json.dumps({"schema_version": 1, "document_id": identity,
        "hashes": hashes}), encoding="utf-8")
    monkeypatch.setattr("papertrace.service_cli.create_service", lambda *args, **kwargs: pytest.fail("corrupt ready must not load"))
    (folder / "chunks.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ImportFailure, match="library_corrupt"):
        PaperLibrary(tmp_path)


def test_processing_marker_blocks_resume_and_lock_blocks_parallel(tmp_path):
    """未完成任务不续跑，同进程不接受并行导入。"""
    identity, folder = _pdf(tmp_path)
    folder.joinpath("processing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ImportFailure, match="import_already_attempted"):
        PaperLibrary(tmp_path).import_document(identity)
    lib = PaperLibrary(tmp_path)
    lib._import_lock.acquire()
    try:
        with pytest.raises(ImportFailure, match="import_busy"):
            lib.import_document(identity)
    finally:
        lib._import_lock.release()


def test_empty_text_and_page_limits_are_classified(tmp_path, monkeypatch):
    """没有可提取文本时不能发布论文。"""
    identity, folder = _pdf(tmp_path)

    class Reader:
        is_encrypted = False
        pages = [object()]

    monkeypatch.setattr(library, "PdfReader", lambda path: Reader())
    monkeypatch.setattr(library, "extract_pdf_pages", lambda path: {"error": None, "pages": []})
    with pytest.raises(ImportFailure, match="no_extractable_text"):
        PaperLibrary(tmp_path).import_document(identity)


@pytest.mark.parametrize("encrypted,pages,expected", [(True, 1, "encrypted_pdf"),
    (False, 61, "page_limit"), (False, 0, "page_limit")])
def test_pdf_reader_limits(tmp_path, monkeypatch, encrypted, pages, expected):
    """加密和超页数在解析、向量化之前拒绝。"""
    identity, _ = _pdf(tmp_path)
    from types import SimpleNamespace
    Reader = lambda: SimpleNamespace(is_encrypted=encrypted, pages=[None] * pages)
    monkeypatch.setattr(library, "PdfReader", lambda path: Reader())
    with pytest.raises(ImportFailure, match=expected):
        PaperLibrary(tmp_path).import_document(identity)


def _text_pdf(tmp_path):
    """生成一页带文本的最小PDF，供真实解析和分块离线联调。"""
    from io import BytesIO
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 30 300 Td (The model uses attention.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return _pdf(tmp_path, buffer.getvalue())


def test_real_parse_chunk_vector_restore_and_cached_qa(tmp_path, monkeypatch):
    """真实PDF解析、标准向量快照、缓存和混合问答只用模拟模型。"""
    from papertrace.embedding_cache import embed_chunks
    from papertrace.generation import GenerationResult
    calls = []
    class Embedder:
        model = "BAAI/bge-m3"
        def embed(self, texts):
            """记录离线向量请求并生成合法非零向量。"""
            calls.append(list(texts))
            return [[1.] + [0.] * 1023 for _ in texts]
    identity, folder = _text_pdf(tmp_path)
    lib = PaperLibrary(tmp_path, embedder_factory=Embedder, offline_query_cache=True)
    assert lib.import_document(identity)["chunk_count"] == 1
    assert len(calls) == 1
    assert lib.import_document(identity)["chunk_count"] == 1
    assert len(calls) == 1
    restored = PaperLibrary(tmp_path, offline_query_cache=True)
    service = restored.service(identity)
    embed_chunks([{"chunk_id": "q", "document_id": "q", "page_number": 1, "text": "attention"}],
                 tmp_path / "query-cache", embedder_factory=Embedder)
    class Generator:
        def generate(self, messages, **kwargs):
            """只返回已有候选的引用，不访问真实模型。"""
            return GenerationResult(json.dumps({"status": "answered", "claims": [{"text": "使用注意力。",
                "citations": [{"segment_id": "E1.S1"}]}], "missing_info": []}), "test", "test", "id", None)
    service.generator = Generator()
    answer = service.answer("attention", identity)
    assert answer["claims"][0]["citations"][0]["document_id"] == identity
    assert answer["validation"]["semantic_support"] == "not_verified"
    assert json.loads((folder / "processing.json").read_bytes())["embedding_batches_attempted"] == 1
    # 不同PDF字节但相同提取文本复用内容缓存，无需再构造向量客户端。
    second, _ = _pdf(tmp_path, (folder / "source.pdf").read_bytes() + b"\n")
    def blocked():
        """全部命中缓存时禁止创建客户端。"""
        pytest.fail("cache hit must not create client")
    assert PaperLibrary(tmp_path, embedder_factory=blocked).import_document(second)["chunk_count"] == 1
    assert len(calls) == 2  # 一次论文向量与一次预填查询向量。


def test_embedding_failure_is_once_and_does_not_publish(tmp_path):
    """向量失败只尝试一批，重启和重复调用都不自动续跑。"""
    calls = []
    class Failing:
        model = "BAAI/bge-m3"
        def embed(self, texts):
            """模拟计费请求失败，不包含可公开异常正文。"""
            calls.append(1)
            raise ValueError("secret")
    identity, folder = _text_pdf(tmp_path)
    with pytest.raises(ImportFailure, match="^import_failed$"):
        PaperLibrary(tmp_path, embedder_factory=Failing).import_document(identity)
    assert calls == [1] and not (folder / "ready.json").exists()
    with pytest.raises(ImportFailure, match="import_already_attempted"):
        PaperLibrary(tmp_path).import_document(identity)


def test_oversize_is_rejected_before_reader(tmp_path, monkeypatch):
    """文件超限在解析前拒绝。"""
    identity, _ = _pdf(tmp_path, b"%PDF-" + b"a" * library.MAX_PDF_BYTES)
    monkeypatch.setattr(library, "PdfReader", lambda path: pytest.fail("unexpected parse"))
    with pytest.raises(ImportFailure, match="invalid_upload"):
        PaperLibrary(tmp_path).import_document(identity)


@pytest.mark.parametrize("kind,expected", [("parse", "pdf_parse_failed"), ("text", "text_limit"),
    ("chunks", "chunk_limit")])
def test_extraction_limits_precede_embedding(tmp_path, monkeypatch, kind, expected):
    """解析失败、文本与分块上限均在真实客户端构造前拒绝。"""
    identity, _ = _text_pdf(tmp_path)
    if kind == "parse":
        monkeypatch.setattr(library, "extract_pdf_pages", lambda path: {"error": "secret", "pages": []})
    elif kind == "text":
        monkeypatch.setattr(library, "extract_pdf_pages", lambda path: {"error": None,
            "pages": [{"cleaned_text": "x" * 300001}]})
    else:
        monkeypatch.setattr(library, "chunk_document", lambda *args, **kwargs: [{}] * 301)
    with pytest.raises(ImportFailure, match=expected):
        PaperLibrary(tmp_path).import_document(identity)
