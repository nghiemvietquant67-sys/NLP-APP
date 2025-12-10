"""
Lab 2: Count Vectorization Test
Test file for CountVectorizer implementation
"""
import sys
sys.path.insert(0, 'c:\\Users\\Quan\\.vscode-R\\NLP-APP')

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def test_count_vectorizer():
    """Test CountVectorizer with sample corpus"""
    
    print("=" * 80)
    print("LAB 2: COUNT VECTORIZATION")
    print("=" * 80)
    
    # Define sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    print("\n[CORPUS]")
    for idx, doc in enumerate(corpus, 1):
        print(f"  Doc {idx}: {doc}")
    
    # Initialize tokenizer and vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)
    
    # Fit and transform
    vectors = vectorizer.fit_transform(corpus)
    
    # Get vocabulary
    vocabulary = vectorizer.vocabulary_
    
    print(f"\n[VOCABULARY] (size: {len(vocabulary)})")
    # Sort by index for better display
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab:
        print(f"  [{idx:2d}] '{token}'")
    
    # Display document-term matrix
    print(f"\n[DOCUMENT-TERM MATRIX]")
    print(f"Shape: {len(vectors)} documents × {len(vocabulary)} features\n")
    
    # Print header with token names
    print("Doc# | " + " | ".join(f"{token:6s}" for token, _ in sorted_vocab))
    print("-" * (8 + 9 * len(vocabulary)))
    
    # Print each document's vector
    for doc_idx, vector in enumerate(vectors, 1):
        row = f"{doc_idx:4d} | " + " | ".join(f"{count:6d}" for count in vector)
        print(row)
    
    # Detailed breakdown
    print(f"\n[DETAILED BREAKDOWN]")
    for doc_idx, (document, vector) in enumerate(zip(corpus, vectors), 1):
        print(f"\nDocument {doc_idx}: \"{document}\"")
        print(f"  Tokens: {tokenizer.tokenize(document)}")
        print(f"  Vector: {vector}")
        
        # Show non-zero counts
        non_zero = [(sorted_vocab[i][0], vector[i]) for i in range(len(vector)) if vector[i] > 0]
        print(f"  Non-zero counts: {non_zero}")
    
    # Test transform on new documents
    print(f"\n{'='*80}")
    print("[TRANSFORM NEW DOCUMENTS]")
    print(f"{'='*80}")
    
    new_docs = [
        "I love AI",
        "NLP and programming"
    ]
    
    new_vectors = vectorizer.transform(new_docs)
    
    for doc, vector in zip(new_docs, new_vectors):
        print(f"\nDocument: \"{doc}\"")
        print(f"  Tokens: {tokenizer.tokenize(doc)}")
        print(f"  Vector: {vector}")
        
        non_zero = [(sorted_vocab[i][0], vector[i]) for i in range(len(vector)) if vector[i] > 0]
        print(f"  Non-zero counts: {non_zero}")
    
    print(f"\n{'='*80}")
    print("✓ CountVectorizer successfully converts text to count vectors!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_count_vectorizer()
