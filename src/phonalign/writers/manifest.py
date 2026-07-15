"""JSON output: one file per utterance plus a corpus-level JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path

from phonalign.align import AlignmentResult


def alignment_record(utt_id: str, wav_path: str, result: AlignmentResult) -> dict:
    return {
        "id": utt_id,
        "wav": str(wav_path),
        "text": result.text,
        "language": result.language,
        "duration": result.audio_duration,
        "phones": [
            {"phone": p.label, "start": p.start, "end": p.end, "score": p.score}
            for p in result.phones
        ],
        "words": [{"word": w.label, "start": w.start, "end": w.end} for w in result.words],
    }


class ManifestWriter:
    """Writes per-utterance JSON files and appends to manifest.jsonl."""

    def __init__(self, out_dir: str | Path):
        self.json_dir = Path(out_dir) / "json"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(out_dir) / "manifest.jsonl"
        self._fh = open(self.manifest_path, "w", encoding="utf-8")

    def add(self, utt_id: str, wav_path: str, result: AlignmentResult) -> None:
        record = alignment_record(utt_id, wav_path, result)
        with open(self.json_dir / f"{utt_id}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()
