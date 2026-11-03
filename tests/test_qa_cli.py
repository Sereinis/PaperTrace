import json
from copy import deepcopy

import pytest

from papertrace import qa_cli
from papertrace.embedding_cache import embedding_config
from papertrace.generation import GenerationResult
from papertrace.qa import QAError
from papertrace.retrieval_eval import corpus_fingerprint


@pytest.fixture(autouse=True)
def no_real_clients(monkeypatch):
    """默认禁止创建真实客户端，确保测试不读取本地配置或调用网络。"""
    def forbidden(**_kwargs):
        """拒绝任何未明确模拟的客户端构造。"""
        pytest.fail("测试禁止创建真实客户端")
    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", forbidden)
    monkeypatch.setattr(qa_cli, "SiliconFlowEmbedder", forbidden)


def _chunks(path):
    payload = {
        "chunks": [{
            "chunk_id": "paper_p001_c001", "document_id": "paper",
            "page_number": 1, "section_key": "abstract", "section_title": "Abstract",
            "text": "The model uses attention.", "char_start": 0, "char_end": 27,
        }],
        "error": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _Generator:
    """为 CLI 离线测试返回固定的合法答案。"""

    def __init__(self, **_kwargs):
        """忽略真实模型配置，避免测试读取本地密钥。"""

    def generate(self, _messages, **_kwargs):
        """返回选择第一个片段的答案。"""
        return GenerationResult(
            content=json.dumps({
                "status": "answered",
                "claims": [{"text": "模型使用注意力。", "citations": [{"segment_id": "E1.S1"}]}],
                "missing_info": [],
            }, ensure_ascii=False),
            requested_model="fake", response_model="fake", request_id="req-1",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def test_main_keyword_json_output(monkeypatch, tmp_path, capsys):
    chunks = tmp_path / "chunks.json"
    _chunks(chunks)
    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", _Generator)
    output = tmp_path / "answer.json"

    assert qa_cli.main([
        "--chunks", str(chunks), "--document-id", "paper",
        "--query", "How does the model work?", "--format", "json",
        "--output", str(output),
    ]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "answered"
    assert result["claims"][0]["citations"][0]["chunk_id"] == "paper_p001_c001"
    assert "问答结果已保存" in capsys.readouterr().out


def test_main_rejects_vector_mode_without_snapshot(monkeypatch, tmp_path, capsys):
    chunks = tmp_path / "chunks.json"
    _chunks(chunks)

    assert qa_cli.main([
        "--chunks", str(chunks), "--document-id", "paper",
        "--query", "question", "--retrieval", "vector",
    ]) == 1
    assert "问答失败" in capsys.readouterr().err


def test_main_human_output(monkeypatch, tmp_path, capsys):
    chunks = tmp_path / "chunks.json"
    _chunks(chunks)
    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", _Generator)

    assert qa_cli.main([
        "--chunks", str(chunks), "--document-id", "paper",
        "--query", "How does the model work?",
    ]) == 0
    output = capsys.readouterr().out
    assert "状态：answered" in output
    assert "paper_p001_c001" in output
    assert "semantic_support=not_verified" in output
    assert "块内字符[" in output
    assert "总计 2 tokens" in output


def _arguments(path, *extra):
    """构建最小合法的命令参数。"""
    return ["--chunks", str(path), "--document-id", "paper", "--query", "attention", *extra]


def _snapshot(path, chunks):
    """写入只含模拟非零向量的标准快照。"""
    payload = {
        "schema_version": 1, "embedding_config": embedding_config(),
        "chunk_count": len(chunks), "chunks_sha256": corpus_fingerprint(chunks),
        "records": [{"chunk": chunk, "embedding": [1.0] + [0.0] * 1023}
                    for chunk in chunks],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_vector_modes_scope_and_parameters(mode, monkeypatch, tmp_path, capsys):
    """两种向量模式使用完整匹配快照，但只在指定论文内检索。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    chunks = json.loads(path.read_text())["chunks"]
    other = dict(chunks[0], chunk_id="other", document_id="other", text="Other attention.")
    chunks.append(other)
    path.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
    vectors = tmp_path / "vectors.json"
    _snapshot(vectors, chunks)
    calls = []

    class Embedder:
        """为实际向量检索器提供模拟查询向量。"""
        def __init__(self, **kwargs):
            """记录模型和无重试配置。"""
            calls.append(kwargs)
            self.model = kwargs["model"]

        def embed(self, texts):
            """每次检索仅向量化一个问题。"""
            assert texts == ["attention"]
            return [[1.0] + [0.0] * 1023]

    monkeypatch.setattr(qa_cli, "SiliconFlowEmbedder", Embedder)
    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", _Generator)
    extra = ["--preserve-source-candidates"] if mode == "hybrid" else []
    assert qa_cli.main(_arguments(path, "--vectors", str(vectors), "--retrieval", mode,
                                  "--format", "json", *extra)) == 0
    result = json.loads(capsys.readouterr().out)
    assert {item["document_id"] for item in result["evidence"]} == {"paper"}
    assert result["retrieval_policy"] == ("source_union" if mode == "hybrid" else "ranked")
    assert calls == [{"model": "BAAI/bge-m3", "max_retries": 0}]


@pytest.mark.parametrize("change", ["missing", "text", "page", "metadata", "boolean"])
def test_snapshot_mismatch_before_clients(change, tmp_path, capsys):
    """同编号但文本、页码或元数据改变也必须在创建客户端前拒绝。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    chunks = json.loads(path.read_text())["chunks"]
    changed = deepcopy(chunks)
    if change == "missing":
        changed = []
    elif change == "text":
        changed[0]["text"] = "Different text."
    elif change == "page":
        changed[0]["page_number"] = 2
    elif change == "metadata":
        changed[0]["extra"] = "changed"
    else:
        changed[0]["char_start"] = False
    vectors = tmp_path / "vectors.json"
    _snapshot(vectors, changed)
    assert qa_cli.main(_arguments(path, "--vectors", str(vectors),
                                  "--retrieval", "hybrid")) == 1
    assert not capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--top-k", "0"], ["--top-k", "21"], ["--max-tokens", "0"],
    ["--max-tokens", "8193"], ["--query", " "], ["--query", "q" * 4001],
    ["--document-id", "missing"], ["--preserve-source-candidates"],
])
def test_invalid_input_before_clients(extra, tmp_path, capsys):
    """非法问题、范围与参数不创建真实客户端，也不打印输入内容。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    assert qa_cli.main(_arguments(path, *extra)) == 1
    assert "问答失败" in capsys.readouterr().err


@pytest.mark.parametrize("payload", [None, [], {"chunks": [] , "error": "SECRET"},
                                     {"chunks": [None]}, {"chunks": [{"text": "SECRET"}]}])
def test_bad_chunk_payload_is_safe(payload, tmp_path, capsys):
    """畸形语料不回显任意正文或异常详情。"""
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert qa_cli.main(_arguments(path)) == 1
    captured = capsys.readouterr()
    assert "SECRET" not in captured.err + captured.out


def test_duplicate_inputs_and_chunk_ids(tmp_path):
    """重复文件以及跨文件重复编号均在调用生成器前失败。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    assert qa_cli.main(["--chunks", str(path), str(path), "--query", "attention",
                        "--document-id", "paper"]) == 1
    other = tmp_path / "other.json"
    _chunks(other)
    assert qa_cli.main(["--chunks", str(path), str(other), "--query", "attention",
                        "--document-id", "paper"]) == 1


def test_no_candidates_requires_no_generation_config(tmp_path, capsys):
    """关键词没有命中时无需读取生成密钥，明确返回候选不足。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    assert qa_cli.main(_arguments(path, "--query", "unrelated", "--format", "json")) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "insufficient" and result["generation"] is None


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_existing_and_unusable_output_preflight(fmt, tmp_path):
    """已有目标或不可用父路径在计费前失败，不破坏原文件。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    output = tmp_path / "existing"
    output.write_text("KEEP", encoding="utf-8")
    assert qa_cli.main(_arguments(path, "--output", str(output), "--format", fmt)) == 1
    assert output.read_text() == "KEEP"
    assert qa_cli.main(_arguments(path, "--output", str(output / "child"), "--format", fmt)) == 1
    assert qa_cli.main(_arguments(path, "--output", str(path), "--format", fmt)) == 1


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_concurrent_output_does_not_overwrite(fmt, monkeypatch, tmp_path):
    """生成期间并发创建的目标在发布时仍不会被替换。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    output = tmp_path / "answer"

    class RacingGenerator(_Generator):
        """在生成期间模拟另一个进程创建输出。"""
        def generate(self, *args, **kwargs):
            """创建应受保护的目标，再返回合法答案。"""
            output.write_text("KEEP", encoding="utf-8")
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", RacingGenerator)
    assert qa_cli.main(_arguments(path, "--output", str(output), "--format", fmt)) == 1
    assert output.read_text() == "KEEP"
    assert not list(tmp_path.glob(".papertrace-qa-*"))


def test_failure_preserves_safe_usage_no_output(monkeypatch, tmp_path, capsys):
    """生成失败只报告净化后的诊断和用量，不留下半份答案。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    output = tmp_path / "answer.json"

    def fail(*_args, **_kwargs):
        """模拟有合法用量的契约错误。"""
        raise QAError("SECRET", code="answer_contract_invalid",
                      validation_stage="answer_validation", detail_code="fields_invalid",
                      usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})

    monkeypatch.setattr(qa_cli, "answer_question", fail)
    assert qa_cli.main(_arguments(path, "--output", str(output))) == 1
    captured = capsys.readouterr()
    assert "SECRET" not in captured.err and not captured.out
    assert '"total_tokens": 3' in captured.err and "fields_invalid" in captured.err
    assert not output.exists() and not list(tmp_path.glob(".papertrace-qa-*"))


def test_terminal_text_retains_offsets_but_escapes_control_characters():
    """人类可读展示不执行论文或模型输出中的终端控制序列。"""
    result = {"document_id": "paper", "query": "q", "status": "partial",
              "claims": [], "missing_info": ["unknown\x1b[31m\r\x9b"], "generation": {"usage": None}}
    rendered = qa_cli._render_text(result)
    assert "\\u001b" in rendered and "\\u000d" in rendered and "\\u009b" in rendered
    assert "\x1b" not in rendered and "用量未知" in rendered


@pytest.mark.parametrize("status", ["partial", "insufficient"])
def test_text_file_retains_status_and_missing_information(status, monkeypatch, tmp_path):
    """保存展示文本时保留部分或不足状态，不把缺失说明渲染为已答结论。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    output = tmp_path / "nested" / "answer.txt"

    class MissingGenerator(_Generator):
        """模拟有缺口的标准生成结果。"""
        def generate(self, *args, **kwargs):
            """保留或清空结论，并提供必须的缺失说明。"""
            response = super().generate(*args, **kwargs)
            data = json.loads(response.content)
            data.update(status=status, missing_info=["缺少训练用电量依据。"])
            if status == "insufficient":
                data["claims"] = []
            return GenerationResult(content=json.dumps(data), requested_model="fake",
                                    response_model="fake", request_id="req", usage=None)

    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", MissingGenerator)
    assert qa_cli.main(_arguments(path, "--output", str(output))) == 0
    text = output.read_text(encoding="utf-8")
    assert f"状态：{status}" in text and "缺少训练用电量依据。" in text
    assert ("结论 1" in text) == (status == "partial")
    assert "not_verified" in text and not list(output.parent.glob(".papertrace-qa-*"))


def test_model_and_token_limit_passed_once(monkeypatch, tmp_path, capsys):
    """非默认模型与输出上限传给原有适配器，仍仅生成一次。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    calls = []

    class RecordingGenerator(_Generator):
        """记录客户端构造和生成调用。"""
        def __init__(self, **kwargs):
            """保存命令中的模型设置。"""
            calls.append(kwargs)

        def generate(self, messages, **kwargs):
            """检查原有系统提示与固定温度和JSON模式。"""
            assert messages[0]["role"] == "system"
            calls.append(kwargs)
            return super().generate(messages, **kwargs)

    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", RecordingGenerator)
    assert qa_cli.main(_arguments(path, "--model", "deepseek-v4-pro",
                                  "--max-tokens", "512", "--top-k", "1")) == 0
    assert calls == [{"model": "deepseek-v4-pro"},
                     {"max_tokens": 512, "temperature": 0.0, "json_mode": True}]


def test_json_stdout_escapes_c1_without_changing_json(monkeypatch, tmp_path, capsys):
    """JSON终端输出转义C1控制符，解析后仍保留原文本。"""
    path = tmp_path / "chunks.json"
    _chunks(path)
    monkeypatch.setattr(qa_cli, "DeepSeekGenerator", _Generator)
    assert qa_cli.main(_arguments(path, "--query", "attention\x9b", "--format", "json")) == 0
    text = capsys.readouterr().out
    assert "\x9b" not in text and json.loads(text)["query"] == "attention\x9b"
