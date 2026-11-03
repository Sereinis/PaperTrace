"""验证 RRF 融合的名次、元数据和边界行为。"""

import pytest

from papertrace.hybrid_search import HybridRetriever, HybridSearchError, reciprocal_rank_fusion


def row(chunk_id, text="text", score=0.0, rank=1):
    """构造最小的统一检索结果。"""
    return {"chunk_id": chunk_id, "document_id": "doc", "page_number": 1,
            "text": text, "score": score, "rank": rank}


def test_rrf_merges_duplicate_ids_and_uses_stable_ties():
    """同一分块的两路名次累加，分数相同按首次出现顺序稳定排序。"""
    results = reciprocal_rank_fusion(
        [row("a", score=10), row("b", score=9)],
        [row("b", score=.9), row("c", score=.8)], top_k=3, rrf_k=1,
    )
    assert [item["chunk_id"] for item in results] == ["b", "a", "c"]
    assert results[0]["score"] == pytest.approx(1 / 3 + 1 / 2)
    assert [item["rank"] for item in results] == [1, 2, 3]
    assert all(set(item) == {"chunk_id", "document_id", "page_number", "text", "score", "rank"}
               for item in results)


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": True}, {"top_k": 1, "rrf_k": 0}, {"top_k": 1, "rrf_k": True},
])
def test_rrf_rejects_invalid_parameters(kwargs):
    """截断位置和平滑常数必须是正整数。"""
    with pytest.raises(HybridSearchError):
        reciprocal_rank_fusion([], [], **kwargs)


def test_rrf_rejects_inconsistent_duplicate_metadata():
    """两路对同一 chunk 的来源元数据不一致时不能静默覆盖。"""
    with pytest.raises(HybridSearchError):
        reciprocal_rank_fusion([row("a", text="one")], [row("a", text="two")], top_k=1)


def test_rrf_rejects_duplicate_within_one_source():
    """同一路重复返回会虚增名次贡献，必须在融合前拒绝。"""
    with pytest.raises(HybridSearchError):
        reciprocal_rank_fusion([row("a"), row("a")], [], top_k=1)


def test_hybrid_calls_both_retrievers_and_respects_top_k():
    """融合器向两路传递相同 Top-K，并返回统一排名。"""
    class Retriever:
        """记录查询并返回固定结果。"""

        def __init__(self, results):
            """保存固定结果。"""
            self.results, self.calls = results, []

        def search(self, query, top_k):
            """记录参数并返回副本。"""
            self.calls.append((query, top_k))
            return [dict(item) for item in self.results]

    keyword, vector = Retriever([row("a")]), Retriever([row("b")])
    results = HybridRetriever(keyword, vector, rrf_k=60).search("q", top_k=1)
    assert len(results) == 1
    assert keyword.calls == [("q", 1)] and vector.calls == [("q", 1)]


@pytest.mark.parametrize("bad", [None, object()])
def test_hybrid_requires_search_methods(bad):
    """初始化时拒绝没有 search 方法的对象。"""
    with pytest.raises(HybridSearchError):
        HybridRetriever(bad, bad)


def test_union_keeps_single_source_hits_that_final_top_k_drops():
    """两路并集保留单路有效候选，普通搜索结果与旧行为不变。"""
    class Retriever:
        """提供带共享干扰项的合成排名。"""

        def __init__(self, ids):
            """记录各路预设顺序。"""
            self.ids, self.calls = ids, []

        def search(self, query, top_k):
            """每路只返回请求数量且记录调用预算。"""
            self.calls.append((query, top_k))
            return [row(cid, rank=i) for i, cid in enumerate(self.ids[:top_k], 1)]

    first, second = Retriever(["a", "b", "data"]), Retriever(["a", "b", "table"])
    index = HybridRetriever(first, second)
    assert [r["chunk_id"] for r in index.search("q", 3)] == ["a", "b", "data"]
    merged = index.search_candidates("q", 3)
    assert [r["chunk_id"] for r in merged] == ["a", "b", "data", "table"]
    assert [r["rank"] for r in merged] == [1, 2, 3, 4]
    assert first.calls == second.calls == [("q", 3), ("q", 3)]
    merged[0]["text"] = "mutated"
    assert index.search_candidates("q", 3)[0]["text"] == "text"


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_union_invalid_k_before_search(bad):
    """并集入口的非法数量不触发底层检索。"""
    class Forbidden:
        """在非法参数下不应被调用。"""

        def search(self, *args):
            """意外调用直接失败。"""
            pytest.fail("unexpected search")

    with pytest.raises(HybridSearchError):
        HybridRetriever(Forbidden(), Forbidden()).search_candidates("q", bad)


@pytest.mark.parametrize("output", [None, {}, [row("a"), row("b")]])
def test_union_rejects_unbounded_or_malformed_source(output):
    """并集不能绕过单路返回数量和列表契约。"""
    class Source:
        """故意违反单路契约。"""

        def search(self, query, top_k):
            """返回测试注入的畸形结果。"""
            return output

    with pytest.raises(HybridSearchError):
        HybridRetriever(Source(), Source()).search_candidates("q", 1)
