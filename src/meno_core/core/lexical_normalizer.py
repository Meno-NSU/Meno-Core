import re
from typing import List

from nltk import wordpunct_tokenize  # type: ignore[import-untyped]
from nltk.stem.snowball import SnowballStemmer  # type: ignore[import-untyped]

_snow = SnowballStemmer(language="russian")
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def tokenize_for_bm25(text: str) -> List[str]:
    tokens: list[str] = []
    for raw_token in wordpunct_tokenize(text):
        lowered = raw_token.lower()
        normalized = "".join(ch for ch in lowered if ch.isalnum())
        if not normalized:
            continue

        tokens.append(normalized)
        if _CYRILLIC_RE.search(normalized):
            stem = _snow.stem(normalized).strip()
            if stem and stem != normalized:
                tokens.append(stem)

    return tokens


def normalize_for_bm25(text: str) -> str:
    return " ".join(tokenize_for_bm25(text))
