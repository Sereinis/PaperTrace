from copy import deepcopy

import pytest

from papertrace.retrieval_eval import (
    corpus_fingerprint, evaluate_retrieval, score_ranking,
)


@pytest.fixture
def sample():
    """提供两篇论文和包含无命中情况的三个问题。"""
    chunks = [
        {"chunk_id": "a1", "document_id": "a", "page_number": 1, "text": "apple"},
        {"chunk_id": "a2", "document_id": "a", "page_number": 2, "text": "pear"},
        {"chunk_id": "b1", "document_id": "b", "page_number": 1, "text": "apple"},
    ]
    cases = [
        {"query_id": "q1", "document_id": "a", "query": "apple",
         "relevant_chunk_ids": ["a1", "a2"]},
        {"query_id": "q2", "document_id": "a", "query": "unknown",
         "relevant_chunk_ids": ["a1"]},
        {"query_id": "q3", "document_id": "b", "query": "apple",
         "relevant_chunk_ids": ["b1"]},
    ]
    return chunks, cases


def test_metrics_use_all_gold_ids_and_truncated_first_hit():
    """手算多个金标、K 边界和首次命中的指标，区分 Recall 与 Hit。"""
    scores = score_ranking(["x", "a", "b"], ["a", "b"], [3, 1, 2, 3, 5])
    assert scores == {
        "recall@1": 0, "hit@1": 0, "rr@1": 0,
        "recall@2": 0.5, "hit@2": 1, "rr@2": 0.5,
        "recall@3": 1, "hit@3": 1, "rr@3": 0.5,
        "recall@5": 1, "hit@5": 1, "rr@5": 0.5,
    }
    assert score_ranking([], ["a"], [1]) == {
        "recall@1": 0, "hit@1": 0, "rr@1": 0,
    }


@pytest.mark.parametrize("ks", [[], [0], [-1], [1.5], [True]])
def test_invalid_cutoffs(ks):
    """拒绝无效截断位置。"""
    with pytest.raises(ValueError, match="ks"):
        score_ranking([], ["a"], ks)


@pytest.mark.parametrize("retrieved,gold", [
    (["a", "a"], ["a"]), ([], []), ([], ["a", "a"]), ([], [""]),
])
def test_invalid_rankings(retrieved, gold):
    """不让重复排名或空金标产生误导性的指标。"""
    with pytest.raises(ValueError):
        score_ranking(retrieved, gold)


def test_macro_average_scope_and_immutability(sample):
    """无结果问题计零，整体按问题平均，且不跨论文检索或修改输入。"""
    chunks, cases = sample
    original = deepcopy(sample)
    report = evaluate_retrieval(chunks, cases, [1, 5])
    assert report["metrics"]["recall@1"] == pytest.approx(0.5)
    assert report["metrics"]["mrr@1"] == pytest.approx(2 / 3)
    assert report["by_document"]["a"]["metrics"]["recall@1"] == 0.25
    assert report["queries"][0]["results"][0]["chunk_id"] == "a1"
    assert report["queries"][1]["results"] == []
    assert sample == original
    assert report == evaluate_retrieval(chunks, cases, [1, 5])


@pytest.mark.parametrize("target,field,value", [
    ("chunk", "chunk_id", ""), ("chunk", "text", None),
    ("chunk", "page_number", 0), ("case", "query", " "),
    ("case", "document_id", "missing"),
    ("case", "relevant_chunk_ids", []),
    ("case", "relevant_chunk_ids", ["missing"]),
    ("case", "relevant_chunk_ids", ["b1"]),
])
def test_bad_input_is_rejected(sample, target, field, value):
    """未知金标、跨论文金标和畸形字段必须报错。"""
    chunks, cases = sample
    (chunks if target == "chunk" else cases)[0][field] = value
    with pytest.raises(ValueError):
        evaluate_retrieval(chunks, cases)


def test_duplicates_and_empty_inputs(sample):
    """拒绝重复问题、重复 chunk 和空数据集。"""
    chunks, cases = sample
    for bad_chunks, bad_cases in [([], cases), (chunks, []),
                                  (chunks + [chunks[0]], cases),
                                  (chunks, cases + [cases[0]])]:
        with pytest.raises(ValueError):
            evaluate_retrieval(bad_chunks, bad_cases)


def test_evidence_validation(sample):
    """核对证据覆盖范围、页码和原文子串，避免标注静默漂移。"""
    chunks, cases = sample
    cases = cases[2:]
    cases[0]["evidence"] = [{"chunk_id": "b1", "page_number": 1, "quote": "apple"}]
    assert evaluate_retrieval(chunks, cases)["metrics"]["hit@1"] == 1
    for field, value in [("page_number", 2), ("quote", "banana"),
                         ("chunk_id", "a1")]:
        bad = deepcopy(cases)
        bad[0]["evidence"][0][field] = value
        with pytest.raises(ValueError):
            evaluate_retrieval(chunks, bad)


@pytest.mark.parametrize("evidence", [
    [], None, [{"chunk_id": []}],
    [{"chunk_id": "a1", "page_number": 1, "quote": "apple"}],
])
def test_malformed_or_incomplete_evidence(sample, evidence):
    """拒绝畸形证据和未覆盖所有金标的证据列表。"""
    chunks, cases = sample
    cases[0]["evidence"] = evidence
    with pytest.raises(ValueError):
        evaluate_retrieval(chunks, cases)


def test_fingerprint_is_key_order_independent_but_snapshot_sensitive(sample):
    """字典键顺序不影响指纹，语料顺序、内容变化则影响。"""
    chunks, _ = sample
    digest = corpus_fingerprint(chunks)
    assert digest == corpus_fingerprint([dict(reversed(list(c.items()))) for c in chunks])
    assert digest != corpus_fingerprint(list(reversed(chunks)))
    changed = deepcopy(chunks)
    changed[0]["text"] = "new text"
    assert digest != corpus_fingerprint(changed)
