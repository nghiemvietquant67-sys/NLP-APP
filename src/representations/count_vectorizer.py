"""
Lab 2: Count Vectorization Implementation
"""
from src.core.interfaces import Vectorizer, Tokenizer


class CountVectorizer(Vectorizer):
    """
    Count Vectorizer: Bag-of-Words model
    Converts documents to count vectors based on learned vocabulary
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize CountVectorizer with a tokenizer
        
        Args:
            tokenizer: A Tokenizer instance to tokenize documents
        """
        self.tokenizer = tokenizer
        self.vocabulary_ = {}
    
    def fit(self, corpus: list[str]):
        """
        Learn vocabulary from corpus
        
        Args:
            corpus: List of documents
        """
        # Collect all unique tokens
        unique_tokens = set()
        
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
        
        # Create vocabulary: sort tokens and assign indices
        sorted_tokens = sorted(list(unique_tokens))
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}
    
    def transform(self, documents: list[str]) -> list[list[int]]:
        """
        Transform documents to count vectors
        
        Args:
            documents: List of documents to transform
            
        Returns:
            List of count vectors
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
        
        vectors = []
        vocab_size = len(self.vocabulary_)
        
        for document in documents:
            # Create zero vector
            vector = [0] * vocab_size
            
            # Tokenize document
            tokens = self.tokenizer.tokenize(document)
            
            # Count occurrences of each token
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vector[idx] += 1
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """
        Fit and transform in one step
        
        Args:
            corpus: List of documents
            
        Returns:
            List of count vectors
        """
        self.fit(corpus)
        return self.transform(corpus)
