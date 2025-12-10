"""
Lab 1 - Task 2: Regex-based Tokenizer Implementation (Bonus)
"""
import re
from src.core.interfaces import Tokenizer


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
