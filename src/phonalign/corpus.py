"""Corpus ingestion: LJSpeech-style metadata.csv or paired wav+txt folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phonalign.errors import CorpusError

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")
TEXT_EXTS = (".txt", ".lab")


@dataclass
class Utterance:
    id: str
    wav_path: Path
    text: str
    speaker: str | None = None


def _parse_metadata_csv(csv_path: Path, wav_dir: Path, has_speaker: bool) -> list[Utterance]:
    """LJSpeech format: id|text  or  id|text|normalized_text (normalized wins).

    With has_speaker: id|speaker|text.
    """
    utts = []
    for lineno, line in enumerate(
        csv_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            raise CorpusError(f"{csv_path}:{lineno}: expected 'id|text', got {line!r}")
        utt_id = parts[0].strip()
        if has_speaker:
            if len(parts) < 3:
                raise CorpusError(f"{csv_path}:{lineno}: expected 'id|speaker|text'")
            speaker, text = parts[1].strip(), parts[2].strip()
        else:
            speaker = None
            text = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else parts[1].strip()
        wav = _find_audio(wav_dir, utt_id)
        if wav is None:
            raise CorpusError(f"{csv_path}:{lineno}: no audio file for id {utt_id!r} in {wav_dir}")
        utts.append(Utterance(id=utt_id, wav_path=wav, text=text, speaker=speaker))
    return utts


def _find_audio(wav_dir: Path, utt_id: str) -> Path | None:
    for sub in (wav_dir, wav_dir / "wavs"):
        for ext in AUDIO_EXTS:
            p = sub / f"{utt_id}{ext}"
            if p.exists():
                return p
    return None


def _paired_folder(folder: Path) -> list[Utterance]:
    utts = []
    for audio in sorted(folder.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue
        for ext in TEXT_EXTS:
            txt = audio.with_suffix(ext)
            if txt.exists():
                text = txt.read_text(encoding="utf-8").strip()
                if text:
                    utts.append(Utterance(id=audio.stem, wav_path=audio, text=text))
                break
    return utts


def discover(input_path: str | Path, has_speaker: bool = False) -> list[Utterance]:
    """Find utterances under input_path.

    Accepts: a metadata.csv file, a directory containing metadata.csv (wavs in
    ./ or ./wavs), or a directory of audio files with same-name .txt/.lab
    transcripts.
    """
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return _parse_metadata_csv(input_path, input_path.parent, has_speaker)
    if not input_path.is_dir():
        raise CorpusError(f"input path does not exist or is not a dir/csv: {input_path}")
    meta = input_path / "metadata.csv"
    if meta.exists():
        return _parse_metadata_csv(meta, input_path, has_speaker)
    utts = _paired_folder(input_path)
    if not utts:
        raise CorpusError(
            f"no utterances found in {input_path}: expected metadata.csv or "
            f"audio files ({'/'.join(AUDIO_EXTS)}) with matching .txt/.lab transcripts"
        )
    return utts
