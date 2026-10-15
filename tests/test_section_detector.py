import pytest

from papertrace.section_detector import detect_sections


def test_detects_common_unnumbered_headings():
    text = "Abstract\nSummary text.\nIntroduction\nMore text.\nConclusion:"

    assert detect_sections(text) == [
        {
            "section_key": "abstract",
            "title": "Abstract",
            "number": None,
            "line_number": 1,
            "start_char": 0,
            "end_char": 8,
        },
        {
            "section_key": "introduction",
            "title": "Introduction",
            "number": None,
            "line_number": 3,
            "start_char": 23,
            "end_char": 35,
        },
        {
            "section_key": "conclusion",
            "title": "Conclusion",
            "number": None,
            "line_number": 5,
            "start_char": 47,
            "end_char": 58,
        },
    ]


def test_detects_numbered_and_lettered_headings():
    text = "1 Introduction\n3.2 Experimental Setup\nA Additional Ablation Studies"

    sections = detect_sections(text)

    assert [(item["number"], item["section_key"]) for item in sections] == [
        ("1", "introduction"),
        ("3.2", "experiments"),
        ("A", "ablation"),
    ]
    assert sections[1]["title"] == "Experimental Setup"


def test_ignores_sentence_like_lines():
    text = "1. We introduce a model.\n2 Results\nThe model is strong."

    sections = detect_sections(text)

    assert [item["section_key"] for item in sections] == ["results"]


def test_preserves_offsets_and_handles_empty_text():
    text = "  Abstract\n\n  4.1 GLUE"

    assert detect_sections("") == []
    assert detect_sections(text)[0]["start_char"] == 2
    assert detect_sections(text)[1]["start_char"] == 14


def test_offsets_follow_original_crlf_text():
    text = "Body\r\n  2 Methods  \r\nConclusion"
    for item in detect_sections(text):
        fragment = text[item["start_char"]:item["end_char"]]
        assert fragment in {"2 Methods", "Conclusion"}


def test_does_not_classify_keywords_inside_other_words():
    assert detect_sections("2 Remodeling")[0]["section_key"] == "other"


@pytest.mark.parametrize("text", [
    "2014) methods. Pre-trained word embeddings",
    "2018. Contextual string embeddings for sequence",
    "175 billion trainable parameters.1",
    "0.3 F1 behind fine-tuning the entire model. This",
    "A distinctive feature of BERT is its unified ar-",
    "A neural network contains many dense layers",
    "P E(pos,2i) = sin(pos/100002i/dmodel )",
    "N d model dff h d k dv Pdrop",
    "B) = ∥Ui⊤",
    "56 i",
    "Model", "Method", "Models", "Methods",
    "1. Question: Does BERT really need such",
    "2. Question: Does MLM pre-training converge",
    "U S A U S A U S A",
    "A Uj",
])
def test_rejects_real_paper_false_positives(text):
    assert detect_sections(text) == []


@pytest.mark.parametrize(("text", "key"), [
    ("1 I NTRODUCTION", "introduction"),
    ("4 O UR METHOD", "method"),
    ("5 E MPIRICAL EXPERIMENTS", "experiments"),
    ("6 R ELATED WORKS", "related_work"),
    ("8 C ONCLUSION AND FUTURE WORK", "conclusion"),
    ("C.1 Effect of Number of Training Steps", "method"),
    ("C.2 Ablation for Different Masking", "ablation"),
])
def test_classifies_small_caps_and_appendix_subsections(text, key):
    item = detect_sections(text)[0]
    assert item["section_key"] == key
    assert text[item["start_char"]:item["end_char"]] == text
    assert item["title"] == text.split(" ", 1)[1]


def test_keeps_numbered_method_and_uppercase_question_heading():
    assert detect_sections("3 Method")[0]["section_key"] == "method"
    assert detect_sections("3 A REN ’T EXISTING SOLUTIONS GOOD ENOUGH ?")


def test_keeps_known_single_word_appendix_title():
    assert detect_sections("A Appendix")[0]["section_key"] == "appendix"
