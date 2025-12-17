"""Demo script: train a small Word2Vec model on UD_English-EWT train file."""
from gensim.models import Word2Vec
import gensim
import os

from pathlib import Path

DATA = Path("data/UD_English-EWT/en_ewt-ud-train.txt")
OUT = Path("results/word2vec_ewt.model")


def sentence_stream(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # naive split by whitespace (file is tokenized) - keep lowercase
            yield [t.lower() for t in line.split()]


def main():
    if not DATA.exists():
        print("Training file not found:", DATA)
        return

    sentences = sentence_stream(DATA)
    print("Training Word2Vec on EWT (this may take some time)...")
    model = Word2Vec(sentences=sentences, vector_size=50, window=5, min_count=2, workers=1)
    model.save(str(OUT))
    print("Saved model to", OUT)
    print("Most similar to 'computer':", model.wv.most_similar("computer", topn=5))


if __name__ == "__main__":
    main()
