import math

import pytest
import torch

from phonalign.align import AlignmentResult, Phone, PhoneVocabMapper, Word, ctc_forced_align
from phonalign.errors import AlignmentError, UnmappablePhoneError

BLANK = 0


def make_emissions(frame_tokens, vocab_size=6, p=0.9):
    """Emissions where each frame strongly prefers the given token id."""
    T = len(frame_tokens)
    rest = (1.0 - p) / (vocab_size - 1)
    probs = torch.full((T, vocab_size), rest)
    for t, tok in enumerate(frame_tokens):
        probs[t, tok] = p
    return probs.log()


class TestCtcForcedAlign:
    def test_clean_three_token_alignment(self):
        frames = [1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 0]
        spans = ctc_forced_align(make_emissions(frames), torch.tensor([1, 2, 3]), BLANK)
        assert [s[0] for s in spans] == [0, 1, 2]
        starts = [s[1] for s in spans]
        ends = [s[2] for s in spans]
        assert starts == [0, 5, 8]
        assert ends == [3, 8, 10]
        assert all(s[3] > 0.5 for s in spans)

    def test_repeated_token_needs_blank(self):
        frames = [1, 1, 0, 1, 1]
        spans = ctc_forced_align(make_emissions(frames), torch.tensor([1, 1]), BLANK)
        assert len(spans) == 2
        assert spans[0][2] <= spans[1][1]

    def test_empty_targets_raises(self):
        with pytest.raises(AlignmentError):
            ctc_forced_align(make_emissions([0, 0]), torch.tensor([], dtype=torch.long), BLANK)

    def test_too_short_audio_raises(self):
        with pytest.raises(AlignmentError):
            ctc_forced_align(make_emissions([1]), torch.tensor([1, 2, 3]), BLANK)

    def test_spans_cover_all_targets_in_order(self):
        frames = [1, 2, 3, 4, 5, 0]
        spans = ctc_forced_align(
            make_emissions(frames, vocab_size=7), torch.tensor([1, 2, 3, 4, 5]), BLANK
        )
        assert [s[0] for s in spans] == [0, 1, 2, 3, 4]


class TestPhoneVocabMapper:
    VOCAB = {"<pad>": 0, "a": 1, "b": 2, "aː": 3, "tʃ": 4, "t": 5, "ʃ": 6, "k": 7}

    def setup_method(self):
        self.mapper = PhoneVocabMapper(self.VOCAB)

    def test_exact(self):
        assert self.mapper.map_phone("a") == [1]
        assert self.mapper.map_phone("aː") == [3]
        assert self.mapper.map_phone("tʃ") == [4]

    def test_stress_stripped(self):
        assert self.mapper.map_phone("ˈa") == [1]

    def test_length_fallback(self):
        # 'bː' not in vocab -> falls back to 'b'
        assert self.mapper.map_phone("bː") == [2]

    def test_greedy_segmentation(self):
        assert self.mapper.map_phone("ab") == [1, 2]
        # aspiration modifier dropped when unmatchable
        assert self.mapper.map_phone("kʰ") == [7]

    def test_tie_bar_stripped(self):
        # t͡ʃ (with tie bar) must hit the single 'tʃ' token, not t+ʃ
        assert self.mapper.map_phone("t͡ʃ") == [4]

    def test_overrides_beat_exact_match(self):
        mapper = PhoneVocabMapper(self.VOCAB, overrides={"a": ["b"], "t": ["t", "ʃ"]})
        assert mapper.map_phone("a") == [2]
        assert mapper.map_phone("t") == [5, 6]
        # non-overridden phones still map normally
        assert mapper.map_phone("b") == [2]

    def test_override_with_missing_token_falls_through(self):
        mapper = PhoneVocabMapper(self.VOCAB, overrides={"a": ["nope"]})
        assert mapper.map_phone("a") == [1]

    def test_unmappable_raises(self):
        with pytest.raises(UnmappablePhoneError):
            self.mapper.map_phone("ɸ", word="x", language="xx")

    def test_fallback_map(self):
        # ɤ (Lao) falls back to ʌ; ASCII g maps to IPA script ɡ
        vocab = dict(self.VOCAB, **{"ʌ": 8, "ɜː": 9, "ɡ": 10})
        mapper = PhoneVocabMapper(vocab)
        assert mapper.map_phone("ɤ") == [8]
        assert mapper.map_phone("ɤː") == [9]
        assert mapper.map_phone("g") == [10]


class TestFullTimeline:
    def make_result(self, phones, duration):
        return AlignmentResult(
            phones=phones, words=[], audio_duration=duration, language="en-us", text=""
        )

    def test_gaps_become_silence(self):
        result = self.make_result(
            [Phone("a", 0.5, 0.6, 1.0), Phone("b", 0.62, 0.7, 1.0)], duration=1.0
        )
        tl = result.full_timeline(min_silence=0.04)
        assert tl[0].label == "sil" and tl[0].start == 0.0
        labels = [p.label for p in tl]
        assert labels == ["sil", "a", "b", "sil"]
        # small 0.02 gap absorbed into 'b'
        assert math.isclose(tl[2].start, 0.6, abs_tol=1e-9)
        # full coverage, no holes
        assert tl[0].start == 0.0 and tl[-1].end == 1.0
        for prev, nxt in zip(tl, tl[1:]):
            assert math.isclose(prev.end, nxt.start, abs_tol=1e-9)

    def test_no_phones_all_silence(self):
        tl = self.make_result([], 2.0).full_timeline()
        assert len(tl) == 1 and tl[0].label == "sil" and tl[0].end == 2.0
