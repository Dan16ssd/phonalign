import numpy as np
import pytest
import soundfile as sf

from phonalign import corpus
from phonalign.errors import CorpusError


def make_wav(path, seconds=0.5, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)


class TestPairedFolder:
    def test_wav_txt_pairs(self, tmp_path):
        make_wav(tmp_path / "a.wav")
        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
        make_wav(tmp_path / "b.wav")
        (tmp_path / "b.lab").write_text("second one", encoding="utf-8")
        make_wav(tmp_path / "orphan.wav")  # no transcript — skipped

        utts = corpus.discover(tmp_path)
        assert [u.id for u in utts] == ["a", "b"]
        assert utts[0].text == "hello world"

    def test_empty_folder_raises(self, tmp_path):
        with pytest.raises(CorpusError):
            corpus.discover(tmp_path)


class TestMetadataCsv:
    def test_ljspeech_layout(self, tmp_path):
        make_wav(tmp_path / "wavs" / "utt1.wav")
        make_wav(tmp_path / "wavs" / "utt2.wav")
        (tmp_path / "metadata.csv").write_text(
            "utt1|Raw text.|Normalized text.\nutt2|Only raw.\n", encoding="utf-8"
        )
        utts = corpus.discover(tmp_path)
        assert utts[0].text == "Normalized text."  # normalized column wins
        assert utts[1].text == "Only raw."

    def test_speaker_column(self, tmp_path):
        make_wav(tmp_path / "utt1.wav")
        (tmp_path / "metadata.csv").write_text("utt1|spk_a|Some text.\n", encoding="utf-8")
        utts = corpus.discover(tmp_path, has_speaker=True)
        assert utts[0].speaker == "spk_a"
        assert utts[0].text == "Some text."

    def test_missing_audio_raises(self, tmp_path):
        (tmp_path / "metadata.csv").write_text("ghost|Text.\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="ghost"):
            corpus.discover(tmp_path)
