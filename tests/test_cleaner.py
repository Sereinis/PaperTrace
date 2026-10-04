from papertrace.cleaner import clean_page_text


def test_normalizes_line_endings():
    text = "first\r\nsecond\rthird"

    assert clean_page_text(text) == "first\nsecond\nthird"


def test_removes_control_characters_but_keeps_line_breaks():
    text = "first\x00 line\nsecond\x07 line"

    assert clean_page_text(text) == "first line\nsecond line"


def test_trims_lines_and_collapses_horizontal_whitespace():
    text = "  first   line  \n\tsecond\t\tline\t"

    assert clean_page_text(text) == "first line\nsecond line"


def test_collapses_repeated_blank_lines():
    text = "first\n\n\n\nsecond"

    assert clean_page_text(text) == "first\n\nsecond"


def test_removes_independent_matching_page_number():
    text = "Results\nThe model performs well.\n8"

    assert clean_page_text(text, page_number=8) == (
        "Results\nThe model performs well."
    )


def test_keeps_other_numbers_and_inline_page_number_values():
    text = "Table 8 reports 28.4 BLEU.\n7\n8 experiments"

    assert clean_page_text(text, page_number=8) == text


def test_keeps_matching_independent_number_inside_page_content():
    text = "A numbered list follows.\n8\nThe next paragraph."

    assert clean_page_text(text, page_number=8) == text


def test_removes_matching_page_number_at_page_start():
    text = "8\nResults\nThe model performs well."

    assert clean_page_text(text, page_number=8) == (
        "Results\nThe model performs well."
    )


def test_keeps_page_number_when_page_number_is_not_provided():
    assert clean_page_text("Results\n8") == "Results\n8"


def test_joins_high_confidence_layout_hyphenation():
    text = "We introduce a language representa-\ntion model."

    assert clean_page_text(text) == (
        "We introduce a language representation model."
    )


def test_keeps_ambiguous_short_hyphenated_words():
    text = "fine-\ntuning\nlow-\nrank"

    assert clean_page_text(text) == text


def test_keeps_compound_word_split_at_line_boundary():
    text = "state-of-the-\nart results"

    assert clean_page_text(text) == text


def test_does_not_join_before_uppercase_word():
    text = "Transformer-\nBased Models"

    assert clean_page_text(text) == text


def test_handles_empty_text():
    assert clean_page_text("") == ""
