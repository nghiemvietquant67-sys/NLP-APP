from __future__ import annotations

import math
from typing import List

import numpy as np

from src.lab_1 import Tokenizer


class TfidfVectorizer:
    """A minimal TF-IDF vectorizer compatible with the simple tokenizers.

    Methods:
        fit(corpus): learns the vocabulary and idf
        transform(docs): transforms docs to TF-IDF numpy arrays
        fit_transform(corpus): fits and transforms
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}
        self.idf_: List[float] = []

    def fit(self, corpus: List[str]) -> None:
        doc_count = len(corpus)
        df: dict[str, int] = {}

        for doc in corpus:
            tokens = set(self.tokenizer.tokenize(doc))
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1

        sorted_tokens = sorted(df.keys())
        self.vocabulary_ = {tok: i for i, tok in enumerate(sorted_tokens)}
        self.idf_ = [math.log((doc_count + 1) / (df[tok] + 1)) + 1.0 for tok in sorted_tokens]

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        vectors = np.zeros((len(documents), len(self.vocabulary_)), dtype=float)

        for i, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            tf: dict[int, int] = {}
            for tok in tokens:
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    tf[idx] = tf.get(idx, 0) + 1

            # compute tf-idf
            for idx, freq in tf.items():
                vectors[i, idx] = (freq / len(tokens)) * self.idf_[idx]

        return vectors

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)


__all__ = ["TfidfVectorizer"]
