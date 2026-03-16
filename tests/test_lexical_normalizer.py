from meno_core.core.lexical_normalizer import normalize_for_bm25, tokenize_for_bm25


def test_tokenize_for_bm25_preserves_latin_digits_and_adds_russian_stems():
    tokens = tokenize_for_bm25("НГУ и students 2025, API-test")

    assert "нгу" in tokens
    assert "students" in tokens
    assert "2025" in tokens
    assert "api" in tokens
    assert "test" in tokens


def test_normalize_for_bm25_returns_space_joined_tokens():
    normalized = normalize_for_bm25("ФИТ НГУ")

    assert isinstance(normalized, str)
    assert "фит" in normalized
    assert "нгу" in normalized
