"""
Lab 1: Tokenization - consolidated implementations

This module combines the SimpleTokenizer and RegexTokenizer
implementations for the lab exercises.
"""
from __future__ import annotations

import re
from typing import List

from src.core.interfaces import Tokenizer


class SimpleTokenizer(Tokenizer):
    """Simple tokenizer that lowercases text, splits on whitespace and
    separates basic punctuation from words.
    """

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()

        tokens: List[str] = []
        # Basic punctuation to split out
        punctuation = r"[.,:;!?\"'()-]"

        for word in text.split():
            if re.search(punctuation, word):
                parts = re.split(f"({punctuation})", word)
                parts = [p for p in parts if p]
                tokens.extend(parts)
            else:
                tokens.append(word)

        return tokens


class RegexTokenizer(Tokenizer):
    """Regex-based tokenizer using a single pattern to extract tokens.

    Pattern: \w+|[^\w\s]
    """

    def __init__(self) -> None:
        self.pattern = r"\w+|[^\w\s]"

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(self.pattern, text)
        return tokens


__all__ = ["SimpleTokenizer", "RegexTokenizer"]


class CountVectorizer:
    """Simple CountVectorizer (Bag-of-Words) that uses a Tokenizer instance."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}

    def fit(self, corpus: list[str]) -> None:
        unique_tokens: set[str] = set()
        for doc in corpus:
            toks = self.tokenizer.tokenize(doc)
            unique_tokens.update(toks)
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {tok: i for i, tok in enumerate(sorted_tokens)}

    def transform(self, documents: list[str]) -> list[list[int]]:
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
        vectors: list[list[int]] = []
        vocab_size = len(self.vocabulary_)
        for doc in documents:
            vec = [0] * vocab_size
            for tok in self.tokenizer.tokenize(doc):
                idx = self.vocabulary_.get(tok)
                if idx is not None:
                    vec[idx] += 1
            vectors.append(vec)
        return vectors

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)

__all__.extend(["CountVectorizer"])
"""
Lab 1: Text Tokenization
Tokenizer implementations: SimpleTokenizer and RegexTokenizer
"""
import re
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into a list of tokens"""
        pass


class SimpleTokenizer(Tokenizer):
    """
    Simple tokenizer that:
    1. Converts text to lowercase
    2. Splits by whitespace
    3. Handles basic punctuation
    """
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using simple whitespace splitting and punctuation handling
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
            
        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokenizer.tokenize("Hello, world!")
            ['hello', ',', 'world', '!']
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split by whitespace first
        words = text.split()
        
        tokens = []
        punctuation = r'[.,:;!?\"\'-]'
        
        for word in words:
            # Check if word contains punctuation
            if re.search(punctuation, word):
                # Split word and punctuation
                parts = re.split(f'({punctuation})', word)
                # Filter out empty strings
                parts = [p for p in parts if p]
                tokens.extend(parts)
            else:
                tokens.append(word)
        
        return tokens


class RegexTokenizer(Tokenizer):
    """
    Regex-based tokenizer using a single pattern to extract tokens.
    Pattern: \w+|[^\w\s]
    - \w+ : matches one or more word characters (alphanumeric + underscore)
    - [^\w\s] : matches any single character that is not a word character or whitespace (punctuation)
    """
    
    def __init__(self):
        """Initialize the regex pattern"""
        # Pattern to match words or individual punctuation
        self.pattern = r'\w+|[^\w\s]'
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using regex pattern
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
            
        Example:
            >>> tokenizer = RegexTokenizer()
            >>> tokenizer.tokenize("Hello, world!")
            ['hello', ',', 'world', '!']
        """
        # Convert to lowercase
        text = text.lower()
        
        # Use regex to find all tokens
        tokens = re.findall(self.pattern, text)
        
        return tokens
