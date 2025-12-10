from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for tokenizers"""
    
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into a list of tokens
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        pass


class Vectorizer(ABC):
    """Abstract base class for vectorizers"""
    
    @abstractmethod
    def fit(self, corpus: list[str]):
        """
        Learn vocabulary from corpus
        
        Args:
            corpus: List of documents
        """
        pass
    
    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        """
        Transform documents to vectors
        
        Args:
            documents: List of documents
            
        Returns:
            List of count vectors
        """
        pass
    
    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """
        Fit and transform in one step
        
        Args:
            corpus: List of documents
            
        Returns:
            List of count vectors
        """
        pass
