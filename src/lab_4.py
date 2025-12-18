"""Lab 4 consolidated module.

Provides simple tokenizers, a TF-IDF vectorizer, and a TextClassifier
implemented for the lab assignment. This file bundles functionality so
tests and examples can import a single module: `src.lab_4`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text and return a list of tokens."""


class RegexTokenizer(Tokenizer):
    """Tokenizer using a single regex to extract word tokens and punctuation."""

    def __init__(self) -> None:
        self.pattern = r"\w+|[^\w\s]"

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        import re

        return re.findall(self.pattern, text)


class TfidfVectorizer:
    """A minimal TF-IDF vectorizer compatible with the simple tokenizers."""

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
                vectors[i, idx] = (freq / max(1, len(tokens))) * self.idf_[idx]

        return vectors

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)


from sklearn.naive_bayes import MultinomialNB

class TextClassifier:
    """Simple text classifier wrapper using a Vectorizer and LogisticRegression.

    Also supports training an alternative classifier (e.g., Naive Bayes) via
    the `model` keyword when calling `fit`.
    """

    def __init__(self, vectorizer: Any) -> None:
        self.vectorizer = vectorizer
        self._model: LogisticRegression | MultinomialNB | None = None

    def fit(self, texts: List[str], labels: List[int], *, model: str = "logreg") -> None:
        """Train the model.

        Args:
            texts: list of raw texts
            labels: list of integer labels
            model: 'logreg' (default) or 'nb' for MultinomialNB
        """
        X = self.vectorizer.fit_transform(texts)
        X = np.asarray(X)
        if model == "logreg":
            clf = LogisticRegression(solver="liblinear")
        elif model == "nb":
            clf = MultinomialNB()
        else:
            raise ValueError("Unknown model type: %s" % model)

        clf.fit(X, labels)
        self._model = clf

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        X = self.vectorizer.transform(texts)
        X = np.asarray(X)
        preds = self._model.predict(X)
        return [int(p) for p in preds]

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }


__all__ = ["Tokenizer", "RegexTokenizer", "TfidfVectorizer", "TextClassifier"]
