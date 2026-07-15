"""Per-phoneme duration arrays in mel-spectrogram frames.

For duration-supervised TTS (FastSpeech2, VITS duration targets) every mel
frame must be assigned to exactly one phone. Boundaries are rounded
cumulatively (each phone's frame count is the difference of consecutive
rounded boundaries), so the durations always sum exactly to the utterance's
total frame count — no drift regardless of phone count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phonalign.align import AlignmentResult


def total_frames(duration_s: float, sample_rate: int, hop_length: int) -> int:
    """Frame count of a center-padded STFT mel spectrogram (librosa/VITS convention)."""
    return int(duration_s * sample_rate) // hop_length + 1


def phone_durations(
    result: AlignmentResult,
    sample_rate: int = 22050,
    hop_length: int = 256,
    min_silence: float = 0.04,
) -> tuple[list[str], np.ndarray]:
    """Gap-filled phone labels and their integer frame durations."""
    timeline = result.full_timeline(min_silence=min_silence)
    n_frames = total_frames(result.audio_duration, sample_rate, hop_length)
    scale = sample_rate / hop_length
    boundaries = [0]
    for p in timeline[:-1]:
        b = int(round(p.end * scale))
        boundaries.append(min(max(b, boundaries[-1]), n_frames))
    boundaries.append(n_frames)
    durs = np.diff(np.asarray(boundaries, dtype=np.int64))
    labels = [p.label for p in timeline]
    # Drop zero-length entries (possible when a tiny phone rounds to 0 frames
    # would desync labels/durations if kept implicit).
    keep = durs > 0
    if not keep.all():
        labels = [l for l, k in zip(labels, keep) if k]
        durs = durs[keep]
    return labels, durs


def write_durations(
    result: AlignmentResult,
    utt_id: str,
    out_dir: str | Path,
    sample_rate: int = 22050,
    hop_length: int = 256,
    min_silence: float = 0.04,
) -> Path:
    """Write <id>.npy (int64 frame counts) + <id>.json sidecar with labels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, durs = phone_durations(result, sample_rate, hop_length, min_silence)
    npy_path = out_dir / f"{utt_id}.npy"
    np.save(npy_path, durs)
    sidecar = {
        "phones": labels,
        "total_frames": int(durs.sum()),
        "sample_rate": sample_rate,
        "hop_length": hop_length,
    }
    with open(out_dir / f"{utt_id}.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=2)
    return npy_path
