"""检查仓库内开发集的标注结构，不依赖本地论文或忽略的 chunk 文件。"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest


DATASET = json.loads(
    (Path(__file__).resolve().parents[1] / "data/eval/retrieval-dev.json").read_bytes()
)


def test_reviewed_development_set_identity():
    """固定版本与非独立验证状态，避免将开发标注误记为独立测试。"""
    assert DATASET["schema_version"] == 1
    assert DATASET["dataset_id"] == "papertrace-retrieval-dev-v2"
    assert DATASET["split"] == "development"
    assert DATASET["annotation_status"] == "assistant_reviewed_not_independently_validated"
    assert len({case["query_id"] for case in DATASET["cases"]}) == 12
    assert len(DATASET["cases"]) == 12
    assert Counter(case["document_id"] for case in DATASET["cases"]) == {
        doc["document_id"]: 4 for doc in DATASET["documents"]
    }


def test_document_manifest_and_chunking_are_complete():
    """检查论文来源与固定分块快照声明，不自动替换语料指纹。"""
    documents = DATASET["documents"]
    assert len(documents) == len({doc["document_id"] for doc in documents}) == 3
    for doc in documents:
        assert doc["source_url"].startswith("https://arxiv.org/abs/")
        assert re.fullmatch(r"[0-9a-f]{64}", doc["chunks_sha256"])
    assert DATASET["chunking"] == {
        "max_chars": 1200, "overlap_chars": 100,
        "pypdf_version": "5.9.0", "baseline_commit": "7076ca7",
    }


@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda case: case["query_id"])
def test_each_question_has_one_evidence_entry_per_gold_chunk(case):
    """逐题核对证据唯一性、文档归属和页码；原文子串由实际评测命令验证。"""
    assert case["query"].strip()
    assert case["category"].strip()
    gold = case["relevant_chunk_ids"]
    evidence = case["evidence"]
    assert gold and len(gold) == len(set(gold))
    assert len(evidence) == len(gold)
    assert {entry["chunk_id"] for entry in evidence} == set(gold)
    for entry in evidence:
        match = re.fullmatch(
            re.escape(case["document_id"]) + r"_p(\d{3})_c\d{3}", entry["chunk_id"]
        )
        assert match is not None
        assert type(entry["page_number"]) is int
        assert entry["page_number"] > 0
        assert entry["page_number"] == int(match.group(1))
        assert entry["quote"].strip()
