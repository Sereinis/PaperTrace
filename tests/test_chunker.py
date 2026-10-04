import pytest

from papertrace.chunker import chunk_page


def test_empty_text_returns_no_chunks():
    assert chunk_page("   \n", 1, "paper") == []


def test_short_text_produces_one_chunk_with_metadata():
    chunks = chunk_page("A short sentence.", 8, "Attention Is All You Need")

    assert chunks == [
        {
            "chunk_id": "attention_is_all_you_need_p008_c001",
            "page_number": 8,
            "chunk_index": 1,
            "text": "A short sentence.",
            "char_count": 17,
            "start_char": 0,
            "end_char": 17,
        }
    ]


def test_prefers_sentence_boundary_near_limit():
    text = "First sentence. Second sentence. Third sentence."

    chunks = chunk_page(text, 1, "paper", max_chars=35, overlap_chars=0)

    assert [chunk["text"] for chunk in chunks] == [
        "First sentence. Second sentence.",
        "Third sentence.",
    ]


def test_prefers_newline_boundary_when_no_sentence_boundary_exists():
    text = "table header\nmodel score value\nremaining content"

    chunks = chunk_page(text, 2, "paper", max_chars=31, overlap_chars=0)

    assert chunks[0]["text"] == "table header\nmodel score value"
    assert chunks[1]["text"] == "remaining content"


def test_splits_long_unbroken_text_at_limit():
    text = "x" * 25

    chunks = chunk_page(text, 1, "paper", max_chars=10, overlap_chars=0)

    assert [chunk["char_count"] for chunk in chunks] == [10, 10, 5]
    assert "".join(chunk["text"] for chunk in chunks) == text


def test_overlap_reuses_previous_text_without_splitting_word():
    text = "alpha beta gamma delta epsilon zeta eta theta"

    chunks = chunk_page(text, 1, "paper", max_chars=25, overlap_chars=10)

    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start_char"] < previous["end_char"]
        assert text[current["start_char"]].isalnum()
        assert current["end_char"] > previous["end_char"]


def test_chunk_ids_are_unique_stable_and_ordered():
    text = "One sentence. Two sentence. Three sentence. Four sentence."

    first = chunk_page(text, 7, "BERT paper.pdf", 25, 0)
    second = chunk_page(text, 7, "BERT paper.pdf", 25, 0)

    first_ids = [chunk["chunk_id"] for chunk in first]
    assert first_ids == [chunk["chunk_id"] for chunk in second]
    assert len(first_ids) == len(set(first_ids))
    assert first_ids == [
        f"bert_paper_pdf_p007_c{index:03d}"
        for index in range(1, len(first_ids) + 1)
    ]


def test_all_chunks_keep_page_number_and_respect_limit():
    text = "Sentence with values 28.4 and x². " * 20

    chunks = chunk_page(text, 8, "lora", max_chars=80, overlap_chars=10)

    assert all(chunk["page_number"] == 8 for chunk in chunks)
    assert all(0 < chunk["char_count"] <= 80 for chunk in chunks)
    assert any("28.4" in chunk["text"] for chunk in chunks)
    assert any("x²" in chunk["text"] for chunk in chunks)


def test_character_offsets_reference_original_page_text():
    text = "First result sentence. Second result sentence. Third result sentence."

    chunks = chunk_page(text, 3, "paper", max_chars=35, overlap_chars=8)

    for chunk in chunks:
        assert text[chunk["start_char"] : chunk["end_char"]] == chunk["text"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_number": 0}, "page_number"),
        ({"document_id": "  "}, "document_id"),
        ({"max_chars": 0}, "max_chars"),
        ({"overlap_chars": -1}, "overlap_chars"),
        ({"max_chars": 10, "overlap_chars": 10}, "overlap_chars"),
    ],
)
def test_rejects_invalid_parameters(kwargs, message):
    arguments = {
        "text": "content",
        "page_number": 1,
        "document_id": "paper",
        "max_chars": 1200,
        "overlap_chars": 100,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        chunk_page(**arguments)


def test_large_overlap_cannot_stall_progress():
    text = "one two three four five six seven eight nine ten eleven twelve"

    chunks = chunk_page(text, 1, "paper", max_chars=20, overlap_chars=19)

    assert chunks
    assert len(chunks) < len(text)
    assert chunks[-1]["end_char"] == len(text)
