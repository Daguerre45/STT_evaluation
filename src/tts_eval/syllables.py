import re

_VOWEL_GROUP_RE = re.compile(r"[aeiouyáéíóúâêîôûãõàèìòùü]+", re.IGNORECASE)
_HIATUS_RE = re.compile(r"[aáeéoóâêôãõàèò][íú]|[íú][aáeéoóâêôãõàèò]", re.IGNORECASE)
_CLEAN_WORD_RE = re.compile(r"[^a-zA-ZáéíóúâêîôûãõàèìòùüÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÜçÇñÑ]")


def count_syllables(word: str) -> int:
    """Estimates the syllable count for Spanish (es-ES) and European Portuguese (pt-PT)."""
    cleaned = _CLEAN_WORD_RE.sub("", word.lower())
    if not cleaned:
        return 1
    groups = _VOWEL_GROUP_RE.findall(cleaned)
    if not groups:
        return 1
    count = 0
    for group in groups:
        if _HIATUS_RE.search(group) or len(group) >= 3 and not re.search(r"^[iuü][aeo][iuü]$", group, re.IGNORECASE):
            count += 2
        else:
            count += 1
    return max(1, count)
