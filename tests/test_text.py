from tts_eval.text import sentence_end_word_indexes, tokenize, word_error_counts


def test_normalizes_portuguese_words() -> None:
    assert tokenize("Olá, isto é uma confirmação.") == ["ola", "isto", "e", "uma", "confirmacao"]


def test_ignores_diacritics_for_lexical_comparison() -> None:
    assert tokenize("Hola, como estas?") == tokenize("Hola, ¿cómo estás?")


def test_counts_word_errors() -> None:
    result = word_error_counts(["bom", "dia", "ana"], ["bom", "noite", "ana", "hoje"])
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 1)


def test_detects_sentence_boundaries() -> None:
    assert sentence_end_word_indexes("Bom dia. Como está?") == {1, 3}
