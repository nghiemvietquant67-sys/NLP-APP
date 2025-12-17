"""
Lab 1: Text Tokenization
Test file combining all tokenizer implementations and tests
"""
import sys
sys.path.insert(0, 'c:\\Users\\Quan\\.vscode-R\\NLP-APP')

from src.lab_1 import SimpleTokenizer, RegexTokenizer, CountVectorizer


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
    # --- Lab 2 tests: Count Vectorizer ---
    print("\n" + "=" * 80)
    print("LAB 2: COUNT VECTORIZATION (RUNNING INSIDE lab_1 TEST)")
    print("=" * 80)

    # Reuse RegexTokenizer for vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    vectors = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.vocabulary_

    print(f"\n[VOCABULARY] (size: {len(vocabulary)})")
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab:
        print(f"  [{idx:2d}] '{token}'")

    print(f"\n[DOCUMENT-TERM MATRIX]")
    print(f"Shape: {len(vectors)} documents × {len(vocabulary)} features\n")

    for doc_idx, vector in enumerate(vectors, 1):
        print(f"Doc {doc_idx:2d}: {vector}")

    print("\n✓ Lab 2 test completed (vocabulary & document-term matrix displayed)")
