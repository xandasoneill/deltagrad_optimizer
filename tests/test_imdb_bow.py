import os

import pytest

from deltagrad.data.imdb import build_vocab, IMDBBowDataset, _tokenize


def _make_fake_split(tmp_path, pos_texts, neg_texts):
    split_dir = tmp_path / "split"
    for label, texts in [("pos", pos_texts), ("neg", neg_texts)]:
        class_dir = split_dir / label
        class_dir.mkdir(parents=True)
        for i, text in enumerate(texts):
            (class_dir / f"{i}_0.txt").write_text(text)
    return str(split_dir)


def test_tokenize_strips_br_tags_and_lowercases():
    tokens = _tokenize("Great movie!<br />Loved it's charm.")
    assert "br" not in tokens
    assert "great" in tokens
    assert "it's" in tokens


def test_build_vocab_ranks_by_frequency(tmp_path):
    split_dir = _make_fake_split(
        tmp_path,
        pos_texts=["good good good great", "good great"],
        neg_texts=["bad bad terrible", "bad"],
    )
    vocab = build_vocab(split_dir, vocab_size=3)
    assert vocab["good"] == 0  # most frequent word gets index 0
    assert len(vocab) == 3


def test_dataset_vectorizes_to_expected_counts(tmp_path):
    split_dir = _make_fake_split(tmp_path, pos_texts=["good good great"], neg_texts=["bad"])
    vocab = {"good": 0, "great": 1, "bad": 2}
    dataset = IMDBBowDataset(split_dir, vocab)
    assert len(dataset) == 2

    vec0, label0 = dataset[0]
    assert label0 == 1  # pos split iterated first
    assert vec0.tolist() == [2.0, 1.0, 0.0]

    vec1, label1 = dataset[1]
    assert label1 == 0  # neg split second
    assert vec1.tolist() == [0.0, 0.0, 1.0]


def test_binary_mode_caps_counts_at_one(tmp_path):
    split_dir = _make_fake_split(tmp_path, pos_texts=["good good good"], neg_texts=["bad"])
    vocab = {"good": 0, "bad": 1}
    dataset = IMDBBowDataset(split_dir, vocab, binary=True)
    vec, label = dataset[0]
    assert label == 1
    assert vec.tolist() == [1.0, 0.0]


def test_max_per_class_caps_document_count(tmp_path):
    split_dir = _make_fake_split(
        tmp_path,
        pos_texts=["good"] * 5,
        neg_texts=["bad"] * 5,
    )
    vocab = {"good": 0, "bad": 1}
    dataset = IMDBBowDataset(split_dir, vocab, max_per_class=2)
    assert len(dataset) == 4  # 2 pos + 2 neg, not 5 + 5


@pytest.mark.network
def test_real_download_and_extract(tmp_path):
    from deltagrad.data.imdb import download_and_extract_imdb
    extracted = download_and_extract_imdb(root=str(tmp_path))
    assert os.path.isdir(os.path.join(extracted, "train", "pos"))
    assert os.path.isdir(os.path.join(extracted, "test", "neg"))
