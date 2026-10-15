from copy import deepcopy

import pytest

from papertrace.chunker import chunk_document


def test_separates_sections_and_numbers_chunks_across_segments():
    pages = [{"page_number": 1, "cleaned_text":
              "Paper title\nAbstract\nSummary.\n1 Introduction\nBody."}]
    original = deepcopy(pages)
    chunks = chunk_document(pages, "paper")

    assert [c["section_key"] for c in chunks] == [
        "unknown", "abstract", "introduction"
    ]
    assert [c["text"] for c in chunks] == [
        "Paper title", "Abstract\nSummary.", "1 Introduction\nBody."
    ]
    assert [c["chunk_id"] for c in chunks] == [
        f"paper_p001_c{i:03d}" for i in range(1, 4)
    ]
    assert [c["chunk_index"] for c in chunks] == [1, 2, 3]
    assert chunks[0]["section_title"] is None
    assert chunks[-1]["section_number"] == "1"
    assert pages == original


def test_inherits_section_through_blank_page_and_changes_mid_page():
    pages = [
        {"page_number": 1, "cleaned_text": "1 Introduction\nOpening."},
        {"page_number": 2, "cleaned_text": "  \n"},
        {"page_number": 3, "cleaned_text": "Continuation.\n2 Results\nScores."},
        {"page_number": 4, "cleaned_text": "More scores."},
    ]
    chunks = chunk_document(pages, "paper")
    assert [(c["page_number"], c["section_key"]) for c in chunks] == [
        (1, "introduction"), (3, "introduction"), (3, "results"), (4, "results")
    ]
    assert chunks[-1]["section_title"] == "Results"
    assert chunks[-1]["chunk_index"] == 1


def test_offsets_reference_cleaned_text_including_outer_whitespace():
    text = "  \r\n Abstract\r\n" + "alpha beta gamma delta. " * 10
    text += "\r\n 2 Results\r\n" + "Score sentence. " * 10 + "  \r\n"
    chunks = chunk_document(
        [{"page_number": 7, "cleaned_text": text}], "BERT paper", 50, 10
    )
    boundary = text.index("2 Results")
    for chunk in chunks:
        assert text[chunk["start_char"]:chunk["end_char"]] == chunk["text"]
        assert chunk["char_count"] == len(chunk["text"])
        assert 0 < chunk["char_count"] <= 50
        assert chunk["document_id"] == "BERT paper"
        if chunk["section_key"] == "abstract":
            assert chunk["end_char"] <= boundary
        else:
            assert chunk["section_key"] == "results"
            assert chunk["start_char"] >= boundary
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    abstract = [c for c in chunks if c["section_key"] == "abstract"]
    assert any(b["start_char"] < a["end_char"]
               for a, b in zip(abstract, abstract[1:]))


def test_unknown_title_only_sections_and_independent_documents():
    assert chunk_document([], "paper") == []
    assert chunk_document([{ "page_number": 1, "cleaned_text": " "}], "paper") == []
    headings = chunk_document([
        {"page_number": 1, "cleaned_text": "Abstract\n1 Introduction"}
    ], "paper")
    assert [c["section_key"] for c in headings] == ["abstract", "introduction"]
    unknown = chunk_document([
        {"page_number": 1, "cleaned_text": "Plain content."}
    ], "paper")
    assert unknown[0]["section_key"] == "unknown"


def test_other_heading_does_not_invent_parent_category():
    chunks = chunk_document([
        {"page_number": 1, "cleaned_text": "4 Experiments\n4.1 GLUE\nContent."}
    ], "paper")
    assert chunks[-1]["section_key"] == "other"
    assert chunks[-1]["section_title"] == "GLUE"


@pytest.mark.parametrize("numbers", [[0], [2, 1], [1, 1]])
def test_rejects_nonpositive_or_unordered_pages(numbers):
    with pytest.raises(ValueError, match="page_number"):
        chunk_document([
            {"page_number": number, "cleaned_text": "Text"} for number in numbers
        ], "paper")


@pytest.mark.parametrize("kwargs", [
    {"document_id": ""}, {"document_id": "中文"},
    {"max_chars": 0}, {"overlap_chars": -1},
    {"max_chars": 10, "overlap_chars": 10},
])
def test_validates_parameters_even_with_no_pages(kwargs):
    arguments = {"pages": [], "document_id": "paper"}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        chunk_document(**arguments)


def test_small_character_limit_preserves_every_character():
    chunks = chunk_document([
        {"page_number": 1, "cleaned_text": "abcde"}
    ], "paper", max_chars=1, overlap_chars=0)
    assert [c["text"] for c in chunks] == list("abcde")
