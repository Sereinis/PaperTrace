"""离线检查字段子问题融合、预算和缺失说明隔离，不验证真实语义质量。"""

import copy
import json

import pytest

from papertrace.report_retrieval import FIELD_QUERIES, FieldRetriever
from papertrace.generation import GenerationResult
from papertrace.scope_review import review_field


def item(cid, rank=1, **metadata):
    """生成具有身份和嵌套元数据的合成检索项。"""
    return {"chunk_id": cid, "document_id": "a", "page_number": 1,
            "text": cid, "metadata": {"tag": ["original"]}, "rank": rank, "score": 99, **metadata}


class Retriever:
    """记录各子问题调用，返回可配置候选。"""

    def __init__(self, values):
        """保存合成排名与调用列表。"""
        self.values, self.calls = values, []

    def search(self, query, top_k):
        """按查询原样回放排名，不调用外部客户端。"""
        self.calls.append((query, top_k))
        return self.values[query]


def test_equal_rank_fusion_and_deep_copy():
    """融合不使用底层不同尺度分数，去重并深拷贝。"""
    values = {"q1": [item("a"), item("b", 2)], "q2": [item("c"), item("b", 2)]}
    original = copy.deepcopy(values)
    base = Retriever(values)
    retriever = FieldRetriever(base, ("q1", "q2"))
    result = retriever.search_candidates("ignored untrusted generation task", 2)
    assert [v["chunk_id"] for v in result] == ["b", "a", "c"]
    assert result[0]["score"] == 2 / 62
    assert base.calls == [("q1", 2), ("q2", 2)]
    assert retriever.last_search["source_union"] is False
    assert retriever.last_search["candidate_limit"] == 4
    result[0]["metadata"]["tag"].append("changed")
    assert values == original


def test_union_budget_and_ranked_budget():
    """每个子问题最多底层2K，最终也只保留2K，普通模式为K。"""
    class Union(Retriever):
        """模拟具有并集入口的混合检索器。"""

        def search_candidates(self, query, top_k):
            """为每个子问题生成2K条身份互异的候选。"""
            self.calls.append((query, top_k))
            return [item(query + str(i), i + 1) for i in range(2 * top_k)]

    base = Union({"q1": [item("a")], "q2": [item("b")]})
    retriever = FieldRetriever(base, ("q1", "q2"))
    assert len(retriever.search_candidates("task", 2)) == 4
    assert retriever.last_search["source_union"] is True
    assert [p["candidate_count"] for p in retriever.last_search["queries"]] == [4, 4]
    assert len(retriever.search("task", 1)) == 1
    assert retriever.last_search["source_union"] is False


@pytest.mark.parametrize("queries", [(), ("",), ("a", "a"), ("a",) * 4,
                                          ["a"], (None,), ("x" * 4001,)])
def test_invalid_queries(queries):
    """无效或重复子问题在检索前被拒绝。"""
    with pytest.raises(ValueError):
        FieldRetriever(Retriever({}), queries)


@pytest.mark.parametrize("top_k", [True, 0, 21, 1.5])
def test_invalid_top_k(top_k):
    """深度输入不合法时无底层检索调用。"""
    base = Retriever({})
    with pytest.raises(ValueError):
        FieldRetriever(base, ("q",)).search("task", top_k)
    assert base.calls == []


@pytest.mark.parametrize("values", [
    {"q1": [item("a"), item("a")], "q2": []},
    {"q1": [item("a")], "q2": [item("a", text="changed")]},
    {"q1": [item("a"), item("b"), item("c")], "q2": []},
    {"q1": None, "q2": []},
])
def test_invalid_results(values):
    """重复ID、冲突元数据、超预算或非列表候选失败闭合。"""
    with pytest.raises(ValueError):
        FieldRetriever(Retriever(values), ("q1", "q2")).search_candidates("task", 2)


def test_empty_candidates():
    """全空候选无伪造分数或证据。"""
    r = FieldRetriever(Retriever({"q": []}), ("q",))
    assert r.search_candidates("task") == []
    assert r.last_search["final_candidate_count"] == 0


def test_balanced_prefix_retains_unique_third_rank_over_shared_tail():
    """多路重合尾项不能挤掉各路预留前缀内的独有证据。"""
    values = {q: [item(q + str(i), i + 1) for i in range(3)] +
              [item("shared" + str(i), i + 4) for i in range(5)]
              for q in ("one", "two", "three")}
    class Union(Retriever):
        """模拟底层两路并集，以8条候选重现截断竞争。"""

        def search_candidates(self, query, top_k):
            """原样返回预定排名。"""
            return self.search(query, top_k)

    retriever = FieldRetriever(Union(values), tuple(values))
    result = retriever.search_candidates("task", 5)
    ids = [v["chunk_id"] for v in result]
    assert len(ids) == len(set(ids)) == 10
    assert all(q + "2" in ids for q in values)
    audit = retriever.last_search
    assert audit["reserved_prefix_depth"] == 3
    assert len(audit["reserved_candidate_ids"]) == 9
    assert any(q + "2" not in audit["unconstrained_rrf_candidate_ids"] for q in values)
    assert all(row["retained_candidate_ids"][:3] == row["candidate_ids"][:3] for row in audit["queries"])
    assert all(row["dropped_candidate_ids"] for row in audit["queries"])


def test_reservation_overlap_refills_by_rrf_and_small_budget():
    """重复前缀不消耗多个名额，小预算不越界，空路不会制造证据。"""
    values = {"one": [item("shared"), item("a", 2)],
              "two": [item("shared"), item("b", 2)], "empty": []}
    r = FieldRetriever(Retriever(values), tuple(values))
    result = r.search_candidates("task", 2)
    assert [v["chunk_id"] for v in result] == ["shared", "a", "b"]
    assert r.last_search["reserved_candidate_ids"] == ["shared"]
    assert r.last_search["queries"][-1]["retained_candidate_ids"] == []
    small = FieldRetriever(Retriever({q: rows[:1] for q, rows in values.items()}), tuple(values))
    result = small.search("task", 1)
    assert len(result) == 1 and result[0]["chunk_id"] == "shared"
    assert small.last_search["reserved_prefix_depth"] == 0


def test_single_query_same_as_original_rank_and_audit_is_independent():
    """单路顺序不变，审计元数据不与输入列表共享。"""
    values = {"q": [item(str(i), i + 1) for i in range(5)]}
    r = FieldRetriever(Retriever(values), ("q",))
    assert [x["chunk_id"] for x in r.search_candidates("task", 5)] == list("01234")
    r.last_search["queries"][0]["candidate_ids"].clear()
    assert len(values["q"]) == 5


@pytest.mark.parametrize("top_k", [1, 2, 5, 20])
@pytest.mark.parametrize("union", [False, True])
def test_prefix_membership_and_budget_for_all_supported_depths(top_k, union):
    """不同预算下前缀保留、唯一身份、连续排名和递减分数均成立。"""
    class Bounded(Retriever):
        """为各子问题提供互不重叠且符合入口预算的排名。"""

        def search(self, query, depth):
            """普通入口最多返回K条。"""
            return [item(query + str(i), i + 1) for i in range(depth)]

        def search_candidates(self, query, depth):
            """并集入口最多返回2K条。"""
            return [item(query + str(i), i + 1) for i in range(2 * depth)]

    r = FieldRetriever(Bounded({}), ("a", "b", "c"))
    result = r.search_candidates("task", top_k) if union else r.search("task", top_k)
    limit = (2 if union else 1) * top_k
    ids = [v["chunk_id"] for v in result]
    assert len(ids) == len(set(ids)) == limit
    assert [v["rank"] for v in result] == list(range(1, limit + 1))
    assert [v["score"] for v in result] == sorted((v["score"] for v in result), reverse=True)
    prefix = r.last_search["reserved_prefix_depth"]
    assert all(q + str(i) in ids for q in ("a", "b", "c") for i in range(prefix))


def test_generic_plans_no_case_specific_answers():
    """固定计划不含当前开发论文、答案数字或分块编号，至多三子问题。"""
    text = json.dumps(FIELD_QUERIES)
    assert len(FIELD_QUERIES) == 5 and all(1 <= len(q) <= 3 for q in FIELD_QUERIES.values())
    assert not any(v in text for v in ("Transformer", "27.3", "37000", "p008", "WMT", "P100"))


@pytest.mark.parametrize("top_k", [1, 2, 5, 20])
def test_adaptive_budget_retains_prefix_or_reports_hard_cap(top_k):
    """独立前缀超出2K时扩容；超过40块则公平截断并保留审计。"""
    from papertrace.report_retrieval import field_candidate_budget

    class Union(Retriever):
        """生成三路完全不重叠的有界候选。"""

        def search_candidates(self, query, depth):
            """每路提供2K个候选，模拟最坏的候选竞争。"""
            return [item(query + str(i), i + 1) for i in range(2 * depth)]

    r = FieldRetriever(Union({}), ("a", "b", "c"), adaptive_budget=True)
    result = r.search_candidates("task", top_k)
    audit = r.last_search
    limit = field_candidate_budget(top_k, 3)
    assert len(result) == limit <= 40
    assert audit["required_prefix_candidate_count"] == 3 * top_k
    assert audit["base_candidate_limit"] == 2 * top_k
    assert audit["hard_candidate_limit"] == audit["candidate_limit"] == limit
    assert audit["budget_expanded"] == (limit > 2 * top_k)
    assert audit["prefix_truncated_by_cap"] == (3 * top_k > 40)
    ids = {v["chunk_id"] for v in result}
    dropped = {q + str(i) for q in ("a", "b", "c") for i in range(top_k)} - ids
    assert set(audit["dropped_prefix_candidate_ids"]) == dropped
    if top_k <= 5:
        assert not dropped
        assert audit["reserved_prefix_depth"] == top_k


def test_adaptive_overlap_does_not_expand_without_need():
    """三路前缀重叠时保持2K，不因启用扩容固定填满3K。"""
    values = {q: [item(str(i), i + 1) for i in range(5)] for q in ("a", "b", "c")}
    r = FieldRetriever(Retriever(values), tuple(values), adaptive_budget=True)
    assert len(r.search_candidates("task", 5)) == 5
    assert r.last_search["candidate_limit"] == 10
    assert r.last_search["hard_candidate_limit"] == 15
    assert not r.last_search["budget_expanded"]
    assert r.last_search["reserved_prefix_depth"] == 5
    assert r.last_search["dropped_prefix_candidate_ids"] == []


def test_adaptive_mode_does_not_expand_ranked_entry():
    """只有候选并集入口可扩容，普通搜索保持K的旧契约。"""
    r = FieldRetriever(Retriever({"a": [item("a")], "b": [item("b")], "c": [item("c")]}),
                       ("a", "b", "c"), adaptive_budget=True)
    assert len(r.search("task", 1)) == 1
    assert not r.last_search["budget_expanded"]


@pytest.mark.parametrize("depth,count", [(True, 3), (0, 2), (21, 2), (5, True), (5, 0), (5, 4)])
def test_invalid_budget_inputs(depth, count):
    """预算函数拒绝越界、布尔及非法子问题数。"""
    from papertrace.report_retrieval import field_candidate_budget

    with pytest.raises(ValueError):
        field_candidate_budget(depth, count)


@pytest.mark.parametrize("claims", [[], [{"text": "Target uses shared settings.", "citations": []}]])
def test_unverified_missing_is_audit_only(claims):
    """无论有没有结论，未核验缺失正文不进入最终缺口，状态不升级。"""
    class Keep:
        """仅模拟判定，不改变缺失说明或触发真实API。"""

        def generate(self, messages, **kwargs):
            """返回与输入条数相同的合法keep判定。"""
            rows = json.loads(messages[1]["content"])["claims"]
            data = {"decisions": [{"claim_index": i, "verdict": "keep", "reason": "synthetic"}
                                  for i in range(len(rows))]}
            return GenerationResult(json.dumps(data), "test", "test", "id", None)

    qa = {"query": "task", "status": "partial" if claims else "insufficient",
          "claims": claims, "missing_info": ["Other large variant missing 300K steps."],
          "validation": {"semantic_support": "not_verified"}}
    original = copy.deepcopy(qa)
    result = review_field(qa, experiment_scope="small", field="model", generator=Keep(),
                          max_tokens=100, protect_missing=True)
    assert "large" not in json.dumps(result["missing_info"])
    assert result["missing_info_audit"]["original"] == qa["missing_info"]
    assert result["status"] == qa["status"] and qa == original
    result["missing_info_audit"]["original"].append("changed")
    assert qa == original
