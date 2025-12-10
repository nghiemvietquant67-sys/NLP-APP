"""
Lab 1 - Task 1: Simple Tokenizer Implementation
"""
import re
from src.core.interfaces import Tokenizer


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
