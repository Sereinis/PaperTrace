import re
import unicodedata


_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_TRAILING_WORD_FRAGMENT = re.compile(r"([A-Za-z]+)-$")
_LEADING_LOWERCASE_WORD = re.compile(r"^([a-z]+)")


def clean_page_text(text: str, page_number: int | None = None) -> str:
    """Conservatively clean text extracted from one PDF page.

    The function removes layout noise that can be identified with high
    confidence while preserving page-local wording for later evidence
    tracing. It does not rebuild tables, formulas, or cross-page sentences.
    """
    if not text:
        return ""

    normalized = _normalize_line_endings(text)
    normalized = _remove_control_characters(normalized)
    lines = [_clean_line(line) for line in normalized.split("\n")]
    lines = _remove_matching_page_number(lines, page_number)
    lines = _join_layout_hyphenation(lines)
    lines = _collapse_blank_lines(lines)
    return "\n".join(lines).strip()


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _remove_control_characters(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in {"\n", "\t"}
        or unicodedata.category(character) != "Cc"
    )


def _clean_line(line: str) -> str:
    return _HORIZONTAL_WHITESPACE.sub(" ", line).strip()


def _remove_matching_page_number(
    lines: list[str], page_number: int | None
) -> list[str]:
    if page_number is None:
        return lines

    expected_page_number = str(page_number)
    cleaned = list(lines)
    non_blank_indexes = [
        index for index, line in enumerate(cleaned) if line
    ]
    if not non_blank_indexes:
        return cleaned

    first_index = non_blank_indexes[0]
    last_index = non_blank_indexes[-1]
    if cleaned[first_index] == expected_page_number:
        cleaned[first_index] = ""
    if cleaned[last_index] == expected_page_number:
        cleaned[last_index] = ""
    return cleaned


def _join_layout_hyphenation(lines: list[str]) -> list[str]:
    joined: list[str] = []
    index = 0

    while index < len(lines):
        current = lines[index]
        if index + 1 >= len(lines):
            joined.append(current)
            break

        following = lines[index + 1]
        if _is_high_confidence_layout_hyphenation(current, following):
            joined.append(current[:-1] + following)
            index += 2
            continue

        joined.append(current)
        index += 1

    return joined


def _is_high_confidence_layout_hyphenation(current: str, following: str) -> bool:
    trailing_match = _TRAILING_WORD_FRAGMENT.search(current)
    leading_match = _LEADING_LOWERCASE_WORD.match(following)
    if not trailing_match or not leading_match:
        return False

    fragment = trailing_match.group(1)
    current_word = current.split()[-1]

    # Short fragments such as "fine-" and "low-" are too ambiguous, and a
    # word that already contains a hyphen is likely a genuine compound.
    return len(fragment) >= 6 and current_word.count("-") == 1


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    previous_was_blank = False

    for line in lines:
        is_blank = line == ""
        if is_blank and previous_was_blank:
            continue
        collapsed.append(line)
        previous_was_blank = is_blank

    return collapsed
