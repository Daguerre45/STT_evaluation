import re
import unicodedata
from dataclasses import dataclass

WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ]+(?:['’-][a-zA-ZÀ-ÖØ-öø-ÿ]+)?|\d+", re.UNICODE)
SENTENCE_BREAK_RE = re.compile(r"[.!?;:]+")


def strip_ssml(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def normalize_text(text: str) -> str:
    """Normalize orthography without treating diacritics as spoken-word errors."""
    text = unicodedata.normalize("NFKD", strip_ssml(text)).lower()
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(WORD_RE.findall(text))


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def sentence_end_word_indexes(text: str) -> set[int]:
    """Returns word positions immediately before punctuation that permits a pause."""
    indexes: set[int] = set()
    seen_words = 0
    cursor = 0
    for match in SENTENCE_BREAK_RE.finditer(strip_ssml(text)):
        seen_words += len(WORD_RE.findall(strip_ssml(text)[cursor : match.start()]))
        if seen_words:
            indexes.add(seen_words - 1)
        cursor = match.end()
    return indexes


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class AlignmentOp:
    op: str  # "match", "substitution", "deletion", "insertion"
    ref_word: str | None
    hyp_word: str | None
    ref_index: int | None
    hyp_index: int | None


def align_tokens(reference: list[str], hypothesis: list[str]) -> list[AlignmentOp]:
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j - 1] + cost,
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
            )
    ops: list[AlignmentOp] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] and table[i][j] == table[i - 1][j - 1]:
            ops.append(AlignmentOp("match", reference[i - 1], hypothesis[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + 1:
            ops.append(AlignmentOp("substitution", reference[i - 1], hypothesis[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append(AlignmentOp("deletion", reference[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(AlignmentOp("insertion", None, hypothesis[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def word_error_counts(reference: list[str], hypothesis: list[str]) -> EditCounts:
    """Levenshtein backtrace returning substitutions, deletions and insertions."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            replacement = table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            table[i][j] = min(replacement, table[i - 1][j] + 1, table[i][j - 1] + 1)

    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] and reference[i - 1] == hypothesis[j - 1]:
            i, j = i - 1, j - 1
        elif i and j and table[i][j] == table[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return EditCounts(substitutions, deletions, insertions)
