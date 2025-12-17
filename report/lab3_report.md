# Báo cáo Lab 3 — Word Embeddings (Gensim + PySpark)

**Harito ID:** 2025-09-25  
**Ngày:** 2025-12-17

---

## ✅ Tóm tắt
Bài lab này thực hiện các thí nghiệm embedding từ với **gensim** (dùng model GloVe đã huấn luyện sẵn và huấn luyện Word2Vec trên bộ dữ liệu nhỏ) cùng một demo PySpark để mở rộng huấn luyện Word2Vec. Repository bao gồm:

- `src/representations/word_embedder.py` — lớp `WordEmbedder` tải `glove-wiki-gigaword-50`, cung cấp `get_vector`, `get_similarity`, `get_most_similar`, và `embed_document`. ✅
- `test/lab4_test.py` — test/demo smoke in ra ví dụ (vector của `king`, similarity, từ tương tự, embedding document). ✅
- `test/lab4_embedding_training_demo.py` — huấn luyện một mô hình Word2Vec nhỏ trên `UD_English-EWT` và lưu lại. ✅
- `test/lab4_spark_word2vec_demo.py` — ví dụ PySpark tokenization trên C4 và huấn luyện Word2Vec (nâng cao). ✅
- `notebook/Lab_3_Word_Embeddings.ipynb` — notebook tương tác với ví dụ và hướng dẫn. ✅

**Tiến độ so với checklist:** 10/12 mục đã hoàn thành (phần visualization còn để thêm vào notebook).

---

## Triển khai (những gì đã làm) 🔧
1. Task 1 — Embeddings đã huấn luyện sẵn (Gensim)
   - Triển khai `WordEmbedder` sử dụng `gensim.downloader.load(model_name)`; mặc định `glove-wiki-gigaword-50` (50 chiều).
   - Các phương thức: `get_vector(word)`, `get_similarity(word1, word2)`, `get_most_similar(word, top_n)`.

2. Task 2 — Embedding document
   - Triển khai `embed_document(document, tokenizer)` trung bình các vector từ biết (bỏ qua token OOV) và trả về vector zero nếu không có token hợp lệ.

3. Task 3 — Huấn luyện Word2Vec (gensim)
   - `test/lab4_embedding_training_demo.py` stream văn bản từ `data/UD_English-EWT/en_ewt-ud-train.txt`, huấn luyện Word2Vec nhỏ (`vector_size=50`) và lưu sang `results/word2vec_ewt.model`.

4. Task 4 — Huấn luyện Word2Vec với Spark (nâng cao)
   - `test/lab4_spark_word2vec_demo.py` minh hoạ đọc `data/c4-train...json`, làm sạch & tokenization đơn giản (lowercase, loại dấu câu) và huấn luyện `pyspark.ml.feature.Word2Vec` (vectorSize=100). Đây là demo tối thiểu có thể mở rộng cho dữ liệu lớn hơn.

5. Task 5 — Visualization (PCA / t-SNE)
   - Chưa thêm vào notebook (bước kế tiếp). Tôi đã chuẩn bị snippet code chi tiết trong phần “Cách tái tạo & Visualization” bên dưới để bạn chạy cục bộ.

---

## Cách chạy (tái tạo) ▶️
Yêu cầu:
- Python 3.8+ và pip
- Cài dependencies: `pip install -r requirements.txt` (sẽ cài `gensim`; với job Spark cần thêm `pyspark` nếu chạy demo Spark)
- Lần chạy đầu tiên model pre-trained sẽ tải `glove-wiki-gigaword-50` (~65MB).

Các lệnh / demo:
- Chạy ví dụ smoke (tải model lần đầu):
  ```powershell
  python test/lab4_test.py
  ```
  Kết quả mong đợi: in ra kích thước vector của `king`, giá trị similarity cho `king<>queen` và `king<>man`, top-10 từ tương tự `computer`, và vector embedding của một document.

- Huấn luyện Word2Vec nhỏ với gensim trên UD_English-EWT (demo):
  ```powershell
  python test/lab4_embedding_training_demo.py
  ```
  Kết quả: file `results/word2vec_ewt.model` và một ví dụ các từ tương tự được in ra.

- Chạy demo PySpark Word2Vec (cần Java + pyspark):
  ```powershell
  python test/lab4_spark_word2vec_demo.py
  ```

Notebook (khám phá tương tác):
- Mở `notebook/Lab_3_Word_Embeddings.ipynb` trong Jupyter Lab hoặc VS Code và chạy các cell theo thứ tự.

---

## Kết quả & Phân tích (mong đợi) 📊
- Ví dụ giá trị similarity (xấp xỉ):
  - `sim(king, queen)` thường cao (gần 0.7–0.8 với GloVe-50) — phản ánh quan hệ giới/tước vị.
  - `sim(king, man)` thường thấp hơn một chút so với king<>queen nhưng vẫn cao.
- `get_most_similar('computer')` trả về các từ thuộc lĩnh vực máy tính (ví dụ: 'computers', 'software', 'hardware', 'pc').
- Embedding document (mean của các vector từ) tạo ra vector dày 50 chiều; phù hợp cho các tác vụ similarity và clustering.

Ghi chú về model tự huấn luyện vs model pre-trained:
- Model pre-trained (GloVe) nắm bắt quan hệ ngữ nghĩa rộng rãi từ tập dữ liệu lớn — phù hợp cho ngữ nghĩa tổng quát.
- Model tự huấn luyện (Word2Vec trên EWT) sẽ phản ánh quan hệ đặc thù domain/genre của dữ liệu huấn luyện (hữu ích nếu domain mục tiêu tương đồng) nhưng kém bền vững nếu dữ liệu nhỏ.

---

## Visualization (cách tạo đồ thị cục bộ) ✨
Dùng PCA hoặc t-SNE để giảm vector từ về 2 chiều và vẽ scatter plot.
Ví dụ (chạy trong notebook):

```python
# lấy vectors cho các từ chọn lọc
words = ['king','queen','man','woman','computer','software','apple','orange','bank','river']
vectors = [we.get_vector(w) for w in words]

# PCA (nhanh)
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

# Tùy chọn: t-SNE cho bố cục phi tuyến
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(vectors)
# vẽ tương tự như PCA
```

Diễn giải:
- Những từ có liên quan (ví dụ: `king` & `queen`, `computer` & `software`) thường nhóm gần nhau; cặp từ liên quan tới giới (gender) có thể nằm theo cùng một hướng.

---

## Khó khăn & Giải pháp ⚠️
- Kích thước/tốc độ tải model: model GloVe (~65MB) cần tải lần đầu — ghi chú rõ trong notebook để cảnh báo người dùng.
- OOV tokens: `get_vector` hỗ trợ lookup không phân biệt hoa thường và trả `None` nếu OOV; `embed_document` bỏ qua OOV và trả vector zero nếu không có token hợp lệ.
- Qui mô huấn luyện: huấn luyện Word2Vec trên toàn bộ C4 đòi hỏi tài nguyên phân tán; demo PySpark là bước bắt đầu tối thiểu — để huấn luyện quy mô lớn cần cluster và điều chỉnh `minCount`, `vectorSize`, số worker.

---

## Tests & Kiểm nghiệm ✅
- Chạy `python test/lab4_test.py` để kiểm tra nhanh (in ví dụ).
- Chạy `python test/lab4_embedding_training_demo.py` để huấn luyện và kiểm nghiệm mô hình Word2Vec nhỏ trên UD_English-EWT.
- Chạy `python test/lab4_spark_word2vec_demo.py` trên máy có Java + pyspark để kiểm tra luồng Spark.

---

## Tài liệu tham khảo & đọc thêm 📚
- GloVe: Pennington, Socher, Manning (2014): https://nlp.stanford.edu/projects/glove/  
- Word2Vec: Mikolov et al. (2013): https://arxiv.org/abs/1301.3781  
- Gensim docs: https://radimrehurek.com/gensim/  
- PySpark MLlib Word2Vec docs: https://spark.apache.org/docs/latest/ml-features.html#word2vec

---

## Bước tiếp theo (tuỳ chọn)
- Thêm cell visualization vào `notebook/Lab_3_Word_Embeddings.ipynb` (tôi có thể thêm code PCA & t-SNE cùng ảnh mẫu và commit).
- Thêm phần so sánh ngắn giữa pre-trained và self-trained bằng cách chạy demo self-trained trên EWT và ghi nhận khác biệt vào báo cáo.
- Thêm test smoke CI (lưu ý: có thể cần tải mạng để lấy model pre-trained).

Nếu bạn muốn, tôi có thể (A) thêm code visualization + tạo ảnh mẫu (tôi sẽ commit chúng), và (B) bổ sung phần so sánh sau khi chạy demo self-trained — bạn muốn tôi làm phần nào trước? ✅
