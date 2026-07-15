"""phonalign — audio-to-phoneme forced alignment for TTS dataset preprocessing."""

__version__ = "0.1.0"

from phonalign.align import Aligner, AlignmentResult, Phone, Word

__all__ = ["Aligner", "AlignmentResult", "Phone", "Word", "__version__"]
