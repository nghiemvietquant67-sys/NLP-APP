"""WordEmbedder — loads pre-trained embeddings via gensim and exposes helpers.

Usage:
  from src.representations.word_embedder import WordEmbedder
  we = WordEmbedder('glove-wiki-gigaword-50')
  v = we.get_vector('king')
"""
from __future__ import annotations

import logging
from typing import List

LOGGER = logging.getLogger(__name__)


class WordEmbedder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50") -> None:
        """Load a gensim pre-trained model by name using gensim.downloader.load.
        This will download the model the first time it's used.
        """
        try:
            import gensim.downloader as api
        except Exception as exc:  # pragma: no cover - import behavior depends on env
            LOGGER.exception("gensim not installed")
            raise
        LOGGER.info("Loading embedding model: %s", model_name)
        self.model = api.load(model_name)
        self.dim = self.model.vector_size

    def get_vector(self, word: str):
        """Return embedding vector for a word, or None if OOV."""
        if word in self.model.key_to_index:  # gensim 4+ attribute
            return self.model.get_vector(word)
        # try lowercased
        lw = word.lower()
        if lw in self.model.key_to_index:
            return self.model.get_vector(lw)
        return None

    def get_similarity(self, word1: str, word2: str) -> float:
        """Return cosine similarity between two words; raises KeyError if either OOV."""
        return float(self.model.similarity(word1, word2))

    def get_most_similar(self, word: str, top_n: int = 10) -> List[tuple]:
        """Return list of (word, score) tuples for top_n similar words.
        If word is OOV, raises KeyError.
        """
        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document: str, tokenizer=None):
        """Return document embedding as mean of known word vectors.

        tokenizer: optional callable(text) -> List[str]. If not provided, caller
        may pass a tokenizer from Lab 1 (RegexTokenizer or SimpleTokenizer).
        Returns a list/ndarray (length = model.vector_size).
        If no known tokens -> returns zero vector.
        """
        if tokenizer is None:
            # lazy import to avoid circular imports at module load
            try:
                from src.lab_1 import RegexTokenizer
            except Exception:
                raise RuntimeError("Tokenizer not available; pass tokenizer argument")
            tokenizer = RegexTokenizer()

        tokens = tokenizer.tokenize(document)
        vecs = []
        for tok in tokens:
            v = self.get_vector(tok)
            if v is not None:
                vecs.append(v)
        import numpy as np

        if not vecs:
            return np.zeros(self.dim, dtype=float)
        return np.mean(np.stack(vecs, axis=0), axis=0)
