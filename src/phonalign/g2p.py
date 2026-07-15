"""Grapheme-to-phoneme conversion with a per-language backend registry.

English (and other espeak languages) use phonemizer + espeak-ng; the espeak-ng
shared library is taken from the `espeakng-loader` wheel so no system install
is required. Lao is not supported by espeak-ng, so it uses epitran's
`lao-Laoo` ruleset with laonlp word tokenization (Lao script has no spaces
between words).
"""

from __future__ import annotations

import builtins
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

from phonalign.errors import G2PError


@dataclass
class WordG2P:
    """One input word and the phone sequence G2P produced for it."""

    word: str
    phones: list[str] = field(default_factory=list)


class G2PBackend:
    """Interface: convert text into a list of words with per-word phones."""

    language: str

    def word_phones(self, text: str) -> list[WordG2P]:
        raise NotImplementedError


def ensure_espeak_library() -> str:
    """Point phonemizer at an espeak-ng library. Returns the source used.

    Precedence: an explicitly set PHONEMIZER_ESPEAK_LIBRARY wins, then the
    DLL bundled in the espeakng-loader wheel, then whatever phonemizer finds
    on the system.
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return "env:PHONEMIZER_ESPEAK_LIBRARY"
    try:
        import espeakng_loader
    except ImportError:
        return "system"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
    os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
    return "bundled:espeakng-loader"


@contextmanager
def _utf8_default_open():
    """Force builtins.open to default to UTF-8 while active.

    epitran/panphon open their bundled data files without an encoding
    argument, which breaks on Windows (cp1252 default) unless the interpreter
    runs in UTF-8 mode. Scoped to backend construction only.
    """
    if sys.flags.utf8_mode:
        yield
        return
    import pathlib

    orig_open = builtins.open
    orig_path_open = pathlib.Path.open

    def _open(file, mode="r", *args, **kwargs):
        # encoding is the 2nd positional arg after mode (buffering, encoding, ...)
        encoding_positional = len(args) >= 2
        if "b" not in str(mode) and not encoding_positional and kwargs.get("encoding") is None:
            kwargs["encoding"] = "utf-8"
        return orig_open(file, mode, *args, **kwargs)

    def _path_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        # covers importlib.resources traversables (how panphon reads its data)
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return orig_path_open(self, mode, buffering, encoding, errors, newline)

    builtins.open = _open
    pathlib.Path.open = _path_open
    try:
        yield
    finally:
        builtins.open = orig_open
        pathlib.Path.open = orig_path_open


_PUNCT_STRIP = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


class EspeakG2P(G2PBackend):
    """phonemizer/espeak-ng backend for espeak-supported languages.

    Words are phonemized individually (one word per line) so the word ->
    phones mapping stays exact; espeak merges adjacent function words when
    given whole sentences (e.g. "in the" -> one phone group), which would
    break the word tier.
    """

    def __init__(self, language: str = "en-us", preserve_stress: bool = False):
        ensure_espeak_library()
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        self.language = language
        try:
            self._backend = EspeakBackend(
                language,
                with_stress=preserve_stress,
                preserve_punctuation=False,
            )
        except RuntimeError as exc:
            raise G2PError(f"espeak backend init failed for {language!r}: {exc}") from exc
        self._separator = Separator(word=" ", phone="|")

    def word_phones(self, text: str) -> list[WordG2P]:
        words = []
        for raw in text.split():
            cleaned = _PUNCT_STRIP.sub("", raw)
            if cleaned:
                words.append(cleaned)
        if not words:
            return []
        phonemized = self._backend.phonemize(words, separator=self._separator, strip=True)
        out = []
        for word, phone_str in zip(words, phonemized):
            # espeak can expand one input word to several (numbers, acronyms);
            # keep them as a single WordG2P so word count stays aligned.
            phones = [p for p in re.split(r"[| ]", phone_str) if p]
            if not phones:
                continue
            out.append(WordG2P(word=word, phones=phones))
        return out


#: Lao spellings epitran's lao-Laoo ruleset mishandles; rewritten in the
#: string fed to epitran only (word labels keep the original spelling).
#: - ຫ before a sonorant (incl. the ligature forms ຫຼ ໜ ໝ) is a silent tone-class
#:   marker, but epitran emits it as a spoken /h/ — drop it.
#: - ໃ and ໄ are both /aj/, but epitran only reorders the preposed vowel ໄ
#:   after the initial consonant (ໃນ came out "ajn" instead of "naj").
_LAO_TRANS_NORMALIZE = (
    ("ຫຼ", "ລ"),  # silent ຫ + semivowel-lo ligature
    ("ຫລ", "ລ"),
    ("ໜ", "ນ"),
    ("ຫນ", "ນ"),
    ("ໝ", "ມ"),
    ("ຫມ", "ມ"),
    ("ຫງ", "ງ"),
    ("ຫຍ", "ຍ"),
    ("ຫວ", "ວ"),
    ("ຫຣ", "ຣ"),
    ("ຼ", "ລ"),  # stray semivowel-lo -> ລ
    ("ໃ", "ໄ"),
)


class EpitranLaoG2P(G2PBackend):
    """Lao G2P: laonlp word segmentation + epitran lao-Laoo transliteration."""

    language = "lo"

    def __init__(self):
        with _utf8_default_open():
            import epitran

            self._epi = epitran.Epitran("lao-Laoo")
            from laonlp.tokenize import word_tokenize

        self._tokenize = word_tokenize

    def word_phones(self, text: str) -> list[WordG2P]:
        out = []
        for chunk in text.split():
            for word in self._tokenize(chunk):
                word = word.strip()
                if not word or _PUNCT_STRIP.sub("", word) == "":
                    continue
                trans = word
                for src, dst in _LAO_TRANS_NORMALIZE:
                    trans = trans.replace(src, dst)
                with _utf8_default_open():
                    phones = [p for p in self._epi.trans_list(trans) if p.strip()]
                if phones:
                    out.append(WordG2P(word=word, phones=phones))
        return out


#: Languages with first-class support. Any other espeak-ng language code also
#: works via EspeakG2P (see `phonalign doctor` for the full list).
LANGUAGES = {
    "en-us": "English (US) — espeak-ng",
    "en": "English (GB) — espeak-ng",
    "lo": "Lao — epitran + laonlp",
}


def get_g2p(language: str, preserve_stress: bool = False) -> G2PBackend:
    """Return a G2P backend for the given language code."""
    if language == "lo":
        return EpitranLaoG2P()
    return EspeakG2P(language, preserve_stress=preserve_stress)
