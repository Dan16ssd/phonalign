"""Exception types for phonalign."""


class PhonalignError(Exception):
    """Base class for all phonalign errors."""


class G2PError(PhonalignError):
    """Text could not be converted to phonemes."""


class UnmappablePhoneError(PhonalignError):
    """A phone produced by G2P has no mapping into the acoustic model vocabulary."""

    def __init__(self, phone: str, word: str, language: str):
        self.phone = phone
        self.word = word
        self.language = language
        super().__init__(
            f"Phone {phone!r} (in word {word!r}, language {language!r}) cannot be mapped "
            f"to the acoustic model vocabulary. Add an entry to phonalign.align.FALLBACK_PHONE_MAP "
            f"or report this as a bug."
        )


class AlignmentError(PhonalignError):
    """Forced alignment failed for an utterance."""


class CorpusError(PhonalignError):
    """The input corpus could not be read or has an unrecognized layout."""
