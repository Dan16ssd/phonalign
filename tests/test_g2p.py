import pytest

from phonalign.g2p import EpitranLaoG2P, EspeakG2P, get_g2p


@pytest.fixture(scope="module")
def en_g2p():
    return EspeakG2P("en-us")


@pytest.fixture(scope="module")
def lo_g2p():
    return EpitranLaoG2P()


def test_english_word_count(en_g2p):
    result = en_g2p.word_phones("printing in the only sense")
    assert [w.word for w in result] == ["printing", "in", "the", "only", "sense"]
    assert all(w.phones for w in result)


def test_english_known_phones(en_g2p):
    (word,) = en_g2p.word_phones("hello")
    assert word.phones[0] == "h"
    assert len(word.phones) >= 4


def test_english_punctuation_stripped(en_g2p):
    result = en_g2p.word_phones('"Hello, world!"')
    assert [w.word for w in result] == ["Hello", "world"]


def test_english_empty(en_g2p):
    assert en_g2p.word_phones("  ...  ") == []


def test_lao_basic(lo_g2p):
    result = lo_g2p.word_phones("ສະບາຍດີ")
    assert result, "expected at least one word"
    phones = [p for w in result for p in w.phones]
    assert phones[0] == "s"
    assert "b" in phones and "d" in phones


def test_lao_tokenizes_unspaced_text(lo_g2p):
    # 'I love Lao language' written without spaces — laonlp should split it
    result = lo_g2p.word_phones("ຂ້ອຍຮັກພາສາລາວ")
    assert len(result) >= 2
    assert all(w.phones for w in result)


def test_lao_ligature_normalized(lo_g2p):
    # ຫຼ ligature spelling must not leak the raw combining char U+0EBC
    result = lo_g2p.word_phones("ລົ້ມເຫຼວ")
    phones = [p for w in result for p in w.phones]
    assert "ຼ" not in "".join(phones)
    assert "l" in phones


def test_lao_silent_h_dropped(lo_g2p):
    # ຫ before a sonorant is a tone-class marker, not a spoken /h/
    result = lo_g2p.word_phones("ເຫຼວ ຫຼຽນ ໜ້າ")
    for w in result:
        assert "h" not in w.phones, f"{w.word}: {w.phones}"


def test_lao_preposed_vowel_reordered(lo_g2p):
    # ໃນ is /naj/: the preposed vowel ໃ must come after the initial consonant
    (word,) = lo_g2p.word_phones("ໃນ")
    assert word.phones[0] == "n", word.phones


def test_lao_word_labels_keep_original_spelling(lo_g2p):
    result = lo_g2p.word_phones("ລົ້ມເຫຼວ")
    assert "ຼ" in "".join(w.word for w in result)


def test_registry_dispatch():
    assert isinstance(get_g2p("lo"), EpitranLaoG2P)
    assert isinstance(get_g2p("en-us"), EspeakG2P)
