import pytest

from papertrace.keyword_search import KeywordRetriever


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": "paper_p001_c001",
            "page_number": 1,
            "section_key": "introduction",
            "text": "We introduce a neural machine translation model.",
        },
        {
            "chunk_id": "paper_p005_c001",
            "page_number": 5,
            "section_key": "experiments",
            "text": "Experiments compare BLEU scores on WMT English German.",
        },
        {
            "chunk_id": "paper_p008_c001",
            "page_number": 8,
            "section_key": "results",
            "text": "The model improves BLEU with a lower training cost.",
        },
    ]


def test_returns_relevant_chunks_with_rank_and_score(chunks):
    retriever = KeywordRetriever(chunks)

    results = retriever.search("BLEU training cost", top_k=2)

    assert [result["chunk_id"] for result in results] == [
        "paper_p008_c001",
        "paper_p005_c001",
    ]
    assert [result["rank"] for result in results] == [1, 2]
    assert results[0]["score"] > results[1]["score"] > 0
    assert results[0]["page_number"] == 8
    assert results[0]["section_key"] == "results"


def test_search_is_case_insensitive_and_does_not_mutate_chunks(chunks):
    original = [dict(chunk) for chunk in chunks]
    retriever = KeywordRetriever(chunks)

    results = retriever.search("wMt ENGLISH", top_k=5)

    assert [result["chunk_id"] for result in results] == ["paper_p005_c001"]
    assert chunks == original
    assert "score" not in chunks[0]


def test_unknown_query_and_empty_index_return_no_results():
    assert KeywordRetriever([]).search("anything") == []
    retriever = KeywordRetriever([{"text": "known term"}])
    assert retriever.search("unseen") == []
    assert retriever.search("  !!! ") == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"k1": 0}, "k1"), ({"b": -0.1}, "b"), ({"b": 1.1}, "b")],
)
def test_rejects_invalid_index_parameters(chunks, kwargs, message):
    with pytest.raises(ValueError, match=message):
        KeywordRetriever(chunks, **kwargs)


def test_rejects_invalid_top_k(chunks):
    retriever = KeywordRetriever(chunks)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("model", top_k=0)


def test_tie_breaking_is_stable_by_input_order():
    chunks = [{"chunk_id": "first", "text": "same term"},
              {"chunk_id": "second", "text": "same term"}]
    results = KeywordRetriever(chunks).search("same")
    assert [result["chunk_id"] for result in results] == ["first", "second"]
