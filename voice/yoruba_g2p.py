"""Rule-based Yoruba grapheme-to-phoneme for TTS.

Yoruba spelling is close to phonemic, so a small rule set gets a clean IPA string:
  - ẹ -> ɛ, ọ -> ɔ, ṣ -> ʃ, j -> ɟ, y -> j, r -> ɾ, g -> ɡ, gb -> ɡb, p -> kp
  - vowel + n before a consonant or word end -> nasal vowel (combining tilde)
  - tone after each vowel / syllabic nasal: acute=3 (high), none=2 (mid), grave=1 (low)
Output symbols all exist in Piper's default phoneme id map, so we can fine-tune an existing Piper voice.
"""
from __future__ import annotations

import re
import unicodedata

ACUTE, GRAVE, DOT, TILDE, MACRON = "\u0301", "\u0300", "\u0323", "\u0303", "\u0304"
VOWELS = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "ẹ": "ɛ", "ọ": "ɔ"}
CONS = {"b": "b", "d": "d", "f": "f", "g": "ɡ", "h": "h", "j": "ɟ", "k": "k", "l": "l",
        "m": "m", "n": "n", "p": "kp", "r": "ɾ", "s": "s", "ṣ": "ʃ", "t": "t", "w": "w", "y": "j",
        "c": "k", "v": "v", "z": "z", "x": "ks", "q": "k"}
PUNCT = {",": ",", ".": ".", "?": "?", "!": "!", ";": ";", ":": ":", "-": " "}

_ONES = ["", "ọ̀kan", "èjì", "ẹ̀ta", "ẹ̀rin", "àrún", "ẹ̀fà", "èje", "ẹ̀jọ", "ẹ̀sán", "ẹ̀wá"]


def _graphemes(word: str):
    """Yield (base_letter, tone) units; base keeps the under-dot, tone is 3/2/1 or None."""
    s = unicodedata.normalize("NFD", word)
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        dot, tone = False, None
        while i < len(s) and unicodedata.combining(s[i]):
            m = s[i]
            if m == DOT:
                dot = True
            elif m == ACUTE:
                tone = "3"
            elif m == GRAVE:
                tone = "1"
            elif m == MACRON:
                tone = "2"
            i += 1
        base = unicodedata.normalize("NFC", ch + (DOT if dot else ""))
        yield base, tone


def word_to_ipa(word: str) -> str:
    units = list(_graphemes(word.lower()))
    out = []
    i = 0
    while i < len(units):
        base, tone = units[i]
        nxt = units[i + 1][0] if i + 1 < len(units) else None
        if base in VOWELS:
            v = VOWELS[base]
            # nasal vowel: vowel + n (+ no following vowel)
            if nxt == "n" and (i + 2 >= len(units) or units[i + 2][0] not in VOWELS):
                v += TILDE
                i += 1
            out.append(v + (tone or "2"))
        elif base == "g" and nxt == "b":
            out.append("ɡb"); i += 1
        elif base in ("n", "m") and tone and (nxt is None or nxt not in VOWELS):
            out.append(base + tone)  # syllabic nasal: ń, ǹ, ḿ
        elif base in CONS:
            out.append(CONS[base])
        elif base.isdigit():
            out.append(word_to_ipa(_ONES[int(base)]) if base != "0" else "odo2")
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)          # dataset annotations like [laughter]
    text = text.replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace("—", ", ")
    text = re.sub(r"(\w)'(\w)", r"\1 \2", text)      # elision apostrophe: l'ẹ́sẹ̀ -> l ẹ́sẹ̀
    return re.sub(r"\s+", " ", text).strip()


def to_ipa(text: str) -> str:
    tokens = [t for t in re.split(r"(\s+|[,.?!;:\-])", normalize(text)) if t and not t.isspace()]
    out = []
    for t in tokens:
        if t in PUNCT:
            if out and out[-1] == " ":
                out.pop()
            out.append(PUNCT[t]); out.append(" ")
        else:
            out.append(word_to_ipa(t)); out.append(" ")
    s = "".join(out).strip()
    return s if s.endswith((".", "?", "!")) else s + "."


def to_phonemes(text: str) -> list[str]:
    """List of single-codepoint phonemes, the unit Piper's phoneme_id_map expects."""
    return list(unicodedata.normalize("NFD", to_ipa(text)))


if __name__ == "__main__":
    for s in ["Lọ fá irun rẹ kó tó wù.", "Ìbọn ló bàá l’ẹ́sẹ̀.", "Ẹ kú àárọ̀, ṣé dáadáa ni?", "Ó ń gbọ́ orin."]:
        print(s, "->", to_ipa(s))
