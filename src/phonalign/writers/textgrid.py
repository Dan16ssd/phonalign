"""Praat TextGrid output (word + phone tiers)."""

from __future__ import annotations

from pathlib import Path

from phonalign.align import AlignmentResult


def write_textgrid(result: AlignmentResult, out_path: str | Path) -> Path:
    """Write a TextGrid with 'words' and 'phones' interval tiers."""
    from praatio import textgrid as ptg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_t = result.audio_duration
    word_entries = [(w.start, w.end, w.label) for w in result.words if w.end > w.start]
    phone_entries = [(p.start, p.end, p.label) for p in result.phones if p.end > p.start]

    tg = ptg.Textgrid()
    tg.addTier(ptg.IntervalTier("words", word_entries, minT=0, maxT=max_t))
    tg.addTier(ptg.IntervalTier("phones", phone_entries, minT=0, maxT=max_t))
    tg.save(str(out_path), format="long_textgrid", includeBlankSpaces=True)
    return out_path
