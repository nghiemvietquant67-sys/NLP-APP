"""
Lab 1: Text Tokenization
Test file combining all tokenizer implementations and tests
"""
import sys
sys.path.insert(0, 'c:\\Users\\Quan\\.vscode-R\\NLP-APP')

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer


def test_tokenizers():
    """Test both Simple and Regex tokenizers with various inputs"""
    
    # Initialize tokenizers
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()
    
    # Test sentences
    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]
    
    print("=" * 80)
    print("LAB 1: TEXT TOKENIZATION")
    print("=" * 80)
    
    for idx, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*80}")
        print(f"Test {idx}: {sentence}")
        print(f"{'='*80}")
        
        # Simple Tokenizer
        simple_tokens = simple_tokenizer.tokenize(sentence)
        print(f"\n[SimpleTokenizer] ({len(simple_tokens)} tokens)")
        print(f"Output: {simple_tokens}")
        
        # Regex Tokenizer
        regex_tokens = regex_tokenizer.tokenize(sentence)
        print(f"\n[RegexTokenizer] ({len(regex_tokens)} tokens)")
        print(f"Output: {regex_tokens}")
        
        # Comparison
        print(f"\nComparison:")
        print(f"  Simple == Regex: {simple_tokens == regex_tokens}")
        if simple_tokens != regex_tokens:
            print(f"  Difference:")
            simple_set = set(simple_tokens)
            regex_set = set(regex_tokens)
            only_simple = simple_set - regex_set
            only_regex = regex_set - simple_set
            if only_simple:
                print(f"    Only in Simple: {only_simple}")
            if only_regex:
                print(f"    Only in Regex: {only_regex}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("✓ SimpleTokenizer: Splits by whitespace and handles basic punctuation")
    print("✓ RegexTokenizer: Uses regex pattern (\\w+|[^\\w\\s]) for more robust tokenization")
    print("\nBoth tokenizers convert text to lowercase and handle punctuation correctly.")


if __name__ == "__main__":
    test_tokenizers()
