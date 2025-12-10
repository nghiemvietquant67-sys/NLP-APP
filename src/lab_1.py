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
