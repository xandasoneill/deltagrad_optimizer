import os
import re
import tarfile
import urllib.request
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
_TOKEN_RE = re.compile(r"[a-z']+")


def download_and_extract_imdb(root="data"):
    """Idempotent: skips download/extraction if `root/aclImdb` already exists."""
    extracted_dir = os.path.join(root, "aclImdb")
    if os.path.isdir(extracted_dir):
        return extracted_dir

    os.makedirs(root, exist_ok=True)
    tarball_path = os.path.join(root, "aclImdb_v1.tar.gz")
    if not os.path.isfile(tarball_path):
        urllib.request.urlretrieve(IMDB_URL, tarball_path)

    with tarfile.open(tarball_path) as tar:
        tar.extractall(root)

    return extracted_dir


def _tokenize(text):
    text = text.replace("<br />", " ")
    return _TOKEN_RE.findall(text.lower())


def _iter_split_files(split_dir, max_per_class=None):
    for label, subdir in [(1, "pos"), (0, "neg")]:
        class_dir = os.path.join(split_dir, subdir)
        filenames = sorted(os.listdir(class_dir))
        if max_per_class is not None:
            filenames = filenames[:max_per_class]
        for fname in filenames:
            yield os.path.join(class_dir, fname), label


def build_vocab(train_dir, vocab_size=10_000, max_per_class=None):
    """Word-frequency vocab built from the train split only."""
    counts = Counter()
    for path, _ in _iter_split_files(train_dir, max_per_class):
        with open(path, encoding="utf-8", errors="ignore") as f:
            counts.update(_tokenize(f.read()))
    vocab_words = [word for word, _ in counts.most_common(vocab_size)]
    return {word: i for i, word in enumerate(vocab_words)}


class IMDBBowDataset(Dataset):
    """Bag-of-words IMDB dataset. Tokenizes once at construction time and stores
    sparse {vocab_idx: count} pairs per doc; densifies to a float vector only in
    __getitem__ to keep memory bounded (25k docs x 10k dims densified upfront would
    be roughly 1GB)."""

    def __init__(self, split_dir, vocab, binary=False, max_per_class=None):
        self.vocab_size = len(vocab)
        self.binary = binary
        self.examples = []
        for path, label in _iter_split_files(split_dir, max_per_class):
            with open(path, encoding="utf-8", errors="ignore") as f:
                tokens = _tokenize(f.read())
            counts = Counter(vocab[t] for t in tokens if t in vocab)
            self.examples.append((counts, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        counts, label = self.examples[idx]
        vec = torch.zeros(self.vocab_size)
        for i, c in counts.items():
            vec[i] = 1.0 if self.binary else float(c)
        return vec, label


def get_imdb_bow_loaders(batch_size, vocab_size=10_000, subset_size=None,
                          root="data", num_workers=0):
    """subset_size caps documents *per class* included in the dataset (rather than
    building the full 25k-doc dataset then subsetting) so smoke mode stays fast.
    The vocabulary is always built from the *full* train split regardless, so the
    feature dimensionality stays exactly `vocab_size` -- capping vocab-building
    input too would shrink it to however many unique words the tiny doc sample
    happens to contain, breaking any model built with a fixed input_dim. This is
    cheap (a few seconds) once the corpus is already downloaded/cached locally."""
    extracted_dir = download_and_extract_imdb(root)
    train_dir = os.path.join(extracted_dir, "train")
    test_dir = os.path.join(extracted_dir, "test")

    vocab = build_vocab(train_dir, vocab_size)

    max_per_class = None if subset_size is None else max(1, subset_size // 2)
    trainset = IMDBBowDataset(train_dir, vocab, max_per_class=max_per_class)
    testset = IMDBBowDataset(test_dir, vocab, max_per_class=max_per_class)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader
