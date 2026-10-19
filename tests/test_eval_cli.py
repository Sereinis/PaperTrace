import json

import pytest

from papertrace.eval_cli import main
from papertrace.retrieval_eval import corpus_fingerprint


@pytest.fixture
def cli_inputs(tmp_path):
    """用临时 JSON 文件提供独立于真实 PDF 的评测输入。"""
    chunks = [{"chunk_id": "a1", "document_id": "a", "page_number": 1,
               "text": "apple", "section_key": "results", "start_char": 0}]
    dataset = {
        "schema_version": 1, "dataset_id": "test",
        "documents": [{"document_id": "a", "chunks_sha256": corpus_fingerprint(chunks)}],
        "cases": [{"query_id": "q1", "document_id": "a", "query": "apple",
                   "relevant_chunk_ids": ["a1"]}],
    }
    chunk_file, dataset_file = tmp_path / "chunks.json", tmp_path / "dataset.json"
    chunk_file.write_text(json.dumps({"chunks": chunks, "error": None}), encoding="utf-8")
    dataset_file.write_text(json.dumps(dataset), encoding="utf-8")
    output = tmp_path / "reports" / "report.json"
    argv = ["--chunks", str(chunk_file), "--dataset", str(dataset_file),
            "--output", str(output)]
    return argv, chunk_file, dataset_file, output


def test_report_output_is_deterministic_and_keeps_evidence(cli_inputs):
    """命令创建输出目录，保留证据字段并可重复运行。"""
    argv, _, _, output = cli_inputs
    assert main(argv) == 0
    first = output.read_bytes()
    saved = json.loads(first)
    assert saved["metrics"]["mrr@5"] == 1
    assert saved["queries"][0]["results"][0]["section_key"] == "results"
    assert len(saved["dataset_sha256"]) == 64
    assert main(argv) == 0
    assert output.read_bytes() == first


@pytest.mark.parametrize("mode", [
    "missing", "json", "schema", "gold", "fingerprint", "shape", "nonhex", "manifest",
])
def test_invalid_inputs_do_not_write_report(cli_inputs, capsys, mode):
    """输入文件缺失、畸形、标注错误或指纹变化时明确失败。"""
    argv, chunk_file, dataset_file, output = cli_inputs
    if mode == "missing":
        chunk_file.unlink()
    elif mode == "json":
        chunk_file.write_text("{", encoding="utf-8")
    elif mode == "shape":
        chunk_file.write_text("[]", encoding="utf-8")
    else:
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
        if mode == "schema":
            data["schema_version"] = 2
        elif mode == "gold":
            data["cases"][0]["relevant_chunk_ids"] = ["wrong"]
        elif mode == "nonhex":
            data["documents"][0]["chunks_sha256"] = "z" * 64
        elif mode == "manifest":
            data["documents"][0]["document_id"] = "missing"
        else:
            data["documents"][0]["chunks_sha256"] = "0" * 64
        dataset_file.write_text(json.dumps(data), encoding="utf-8")
    assert main(argv) == 1
    assert "评测失败" in capsys.readouterr().err
    assert not output.exists()


def test_output_cannot_overwrite_input(cli_inputs):
    """拒绝将评测结果覆盖到语料或标注文件。"""
    argv, chunk_file, _, _ = cli_inputs
    original = chunk_file.read_bytes()
    argv[-1] = str(chunk_file)
    assert main(argv) == 1
    assert chunk_file.read_bytes() == original


def test_invalid_k_and_output_directory(cli_inputs):
    """指标参数和输出路径错误均有非零退出码。"""
    argv, _, dataset_file, _ = cli_inputs
    assert main(argv + ["--ks", "0"]) == 1
    argv[-1] = str(dataset_file.parent)
    assert main(argv) == 1
