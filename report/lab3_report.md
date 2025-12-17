# Lab 3 Report — Word Embeddings (Gensim + PySpark)

**Harito ID:** 2025-09-25  
**Ngày:** 2025-12-17

---

## ✅ Summary
This lab implements word embeddings experiments using **gensim** (pre-trained GloVe and training Word2Vec from a small dataset) and a PySpark demo for scaling Word2Vec training. The repository includes:

- `src/representations/word_embedder.py` — WordEmbedder class that loads `glove-wiki-gigaword-50`, provides `get_vector`, `get_similarity`, `get_most_similar`, and `embed_document` methods. ✅
- `test/lab4_test.py` — smoke/demo test that prints examples (king vector, similarities, most-similar words, document embedding). ✅
- `test/lab4_embedding_training_demo.py` — trains a small Word2Vec model on `UD_English-EWT` and saves it. ✅
- `test/lab4_spark_word2vec_demo.py` — PySpark example that tokenizes C4 and trains a Word2Vec model (advanced). ✅
- `notebook/Lab_3_Word_Embeddings.ipynb` — interactive notebook with examples and instructions. ✅

**Progress vs checklist:** 10/12 implementation items completed (visualization steps remain to be added), and report & analysis are documented here.

---

## Implementation (what I did) 🔧
1. Task 1 — Pre-trained embeddings (Gensim)
   - Implemented `WordEmbedder` which loads models via `gensim.downloader.load(model_name)`; default: `glove-wiki-gigaword-50` (50-dimensional).  
   - Implemented methods: `get_vector(word)`, `get_similarity(word1, word2)`, `get_most_similar(word, top_n)`.  

2. Task 2 — Document embedding
   - Implemented `embed_document(document, tokenizer)` that averages known word vectors (ignores OOV tokens) and returns a zero vector if no known tokens are present.

3. Task 3 — Training Word2Vec (gensim)
   - `test/lab4_embedding_training_demo.py` streams text from `data/UD_English-EWT/en_ewt-ud-train.txt`, trains a small Word2Vec model (`vector_size=50`) and saves it to `results/word2vec_ewt.model`.

4. Task 4 — Training Word2Vec with Spark (advanced)
   - `test/lab4_spark_word2vec_demo.py` demonstrates reading `data/c4-train...json`, simple cleaning and tokenization (lowercase, remove punctuation), and trains a `pyspark.ml.feature.Word2Vec` model (vectorSize=100). This is a minimal demo you can expand for larger datasets.

5. Task 5 — Visualization (PCA / t-SNE)
   - Not yet implemented in the notebook (planned next step). I include exact code snippets in the “How to reproduce & visualization” section below so you can run it locally.

---

## How to run (reproduce) ▶️
Prerequisites:
- Python 3.8+ and pip
- Install dependencies: `pip install -r requirements.txt` (adds `gensim`; for Spark jobs add `pyspark` if needed)
- The first run of the pre-trained model will download `glove-wiki-gigaword-50` (~65MB).

Commands / demos:
- Run the smoke examples (downloads model on first run):
  ```powershell
  python test/lab4_test.py
  ```
  Expected outputs: printed vector dimension for `king`, similarity numbers for `king<>queen` and `king<>man`, top-10 words similar to `computer`, and a document embedding vector printed.

- Train a small gensim Word2Vec on UD_English-EWT (demo):
  ```powershell
  python test/lab4_embedding_training_demo.py
  ```
  Output: `results/word2vec_ewt.model` and a small printed sample of most similar words.

- Run the PySpark Word2Vec demo (requires Java + pyspark):
  ```powershell
  python test/lab4_spark_word2vec_demo.py
  ```

Notebook (interactive exploration):
- Open `notebook/Lab_3_Word_Embeddings.ipynb` in Jupyter Lab or VS Code and run cells sequentially.

---

## Results & Analysis (what to expect) 📊
- Example similarity values (approximate):
  - `sim(king, queen)` should be high (close to 0.7–0.8 for GloVe-50) — indicates gender/royalty relation.
  - `sim(king, man)` likely slightly lower than king<>queen, but still high.
- `get_most_similar('computer')` will show words in computing domain (e.g., 'computers', 'software', 'hardware', 'pc').
- Document embeddings (mean of word vectors) produce dense 50-dimensional vectors; they are suitable for downstream similarity checks and clustering.

Notes on Self-trained vs Pre-trained models:
- Pre-trained (GloVe) captures broad semantic relations from large corpora — good general-purpose semantics.
- Self-trained Word2Vec on UD_English-EWT will capture domain/genre-specific associations (useful if target domain is similar to the training set) but may be less robust if training data is small.

---

## Visualization (how to produce plots locally) ✨
Use PCA or t-SNE to reduce word vectors to 2D for scatter plots.
Example (run in notebook):

```python
# get vectors for selected words
words = ['king','queen','man','woman','computer','software','apple','orange','bank','river']
vectors = [we.get_vector(w) for w in words]

# PCA (fast)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(vectors)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(proj[:,0], proj[:,1])
for i,w in enumerate(words):
    plt.text(proj[i,0]+0.01, proj[i,1]+0.01, w)
plt.title('PCA of selected word vectors')
plt.show()

# Optional: t-SNE for non-linear layout
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(vectors)
# plot similar to PCA
```

Interpretation:
- Related words (e.g., `king` & `queen`, `computer` & `software`) should cluster nearby; gender pairs may align along a direction.

---

## Difficulties & Solutions ⚠️
- Model download size/time: the GloVe model (~65MB) downloads on first use — note this in the notebook and warn users.  
- OOV tokens: `get_vector` handles case-insensitive lookup and returns `None` for true OOV; `embed_document` ignores OOV tokens and returns a zero vector if no known tokens found.  
- Training scale: training Word2Vec on the full C4 dataset needs cluster resources; the PySpark demo is a minimal starting point — for large-scale training use distributed cluster resources and adjust `minCount`, `vectorSize`, and workers.

---

## Tests & Validation ✅
- Run `python test/lab4_test.py` for a quick smoke verification (prints examples).  
- Run `python test/lab4_embedding_training_demo.py` to train and validate a small Word2Vec model on UD_English-EWT.  
- Run `python test/lab4_spark_word2vec_demo.py` on a machine with Java + pyspark to test the Spark flow.

---

## References & Further Reading 📚
- GloVe: Pennington, Socher, Manning (2014): https://nlp.stanford.edu/projects/glove/  
- Word2Vec: Mikolov et al. (2013): https://arxiv.org/abs/1301.3781  
- Gensim docs: https://radimrehurek.com/gensim/  
- PySpark MLlib Word2Vec docs: https://spark.apache.org/docs/latest/ml-features.html#word2vec

---

## Next steps (optional)
- Add visualization cells to `notebook/Lab_3_Word_Embeddings.ipynb` (I can add PCA & t-SNE plotting code + sample output).  
- Add a short comparison section that runs pre-trained vs self-trained similarity comparisons and documents differences in the report.  
- Add CI smoke test (note: network download may be required to fetch pretrained model).  

If you want I can now (A) add visualization code + produce sample plots (I will commit them), and (B) append a comparison section after running the self-trained demo on the EWT dataset — which should I do next? ✅
