"""Unit-style smoke test for WordEmbedder.
This test prints expected outputs; it will download gensim model on first run.
"""
import sys

try:
    from src.representations.word_embedder import WordEmbedder
except Exception as exc:
    raise RuntimeError("word_embedder not importable") from exc


def test_word_embedder_examples():
    we = WordEmbedder("glove-wiki-gigaword-50")

    v_king = we.get_vector("king")
    print("king vector dim:", len(v_king))

    sim_kq = we.get_similarity("king", "queen")
    sim_km = we.get_similarity("king", "man")
    print("sim(king,queen)=", sim_kq)
    print("sim(king,man)=", sim_km)

    print("most similar to 'computer':")
    ms = we.get_most_similar("computer", top_n=10)
    for w, s in ms:
        print(" ", w, s)

    doc_vec = we.embed_document("The queen rules the country.")
    print("document vector (queen sentence) dim:", len(doc_vec))


if __name__ == "__main__":
    test_word_embedder_examples()
    print("Done")
