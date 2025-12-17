"""
Lab 1: Tokenization - consolidated implementations

This module provides a minimal, self-contained implementation for the
lab: an abstract Tokenizer base class, two concrete tokenizers
(SimpleTokenizer, RegexTokenizer), and a small CountVectorizer.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text and return a list of tokens."""


class SimpleTokenizer(Tokenizer):
    """Simple whitespace + punctuation tokenizer.

    - Lowers input
    - Splits on whitespace
    - Separates basic punctuation as own tokens
    """

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = []
        punctuation = r"[.,:;!?\"'()\-]"

        for word in text.split():
            if re.search(punctuation, word):
                parts = re.split(f"({punctuation})", word)
                parts = [p for p in parts if p]
                tokens.extend(parts)
            else:
                tokens.append(word)

        return tokens


class RegexTokenizer(Tokenizer):
    """Tokenizer using a single regex to extract word tokens and punctuation."""

    def __init__(self) -> None:
        self.pattern = r"\w+|[^\w\s]"

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(self.pattern, text)


class CountVectorizer:
    """A simple count vectorizer that builds a vocabulary and produces
    document-term count vectors.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}

    def fit(self, corpus: List[str]) -> None:
        unique_tokens = set()
        for doc in corpus:
            unique_tokens.update(self.tokenizer.tokenize(doc))
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {tok: i for i, tok in enumerate(sorted_tokens)}

    def transform(self, documents: List[str]) -> List[List[int]]:
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
        vectors: List[List[int]] = []
        vocab_size = len(self.vocabulary_)
        for doc in documents:
            vec = [0] * vocab_size
            for tok in self.tokenizer.tokenize(doc):
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    vec[idx] += 1
            vectors.append(vec)
        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        self.fit(corpus)
        return self.transform(corpus)


__all__ = ["Tokenizer", "SimpleTokenizer", "RegexTokenizer", "CountVectorizer"]
