## Thông tin cá nhân
- **Họ và tên**: Nghiêm Việt Quân
- **Mã sinh viên (MSV)**: 22001632
- **Lớp**: K67A4


---

## 📌 Lab 1: Tokenization
### Nội dung thực hiện
- Xây dựng **interface `Tokenizer`**.
- Cài đặt **SimpleTokenizer**:
  - Chuyển chữ thường.
  - Tách từ theo khoảng trắng.
  - Xử lý tách dấu câu đơn giản (.,!?).
- Cài đặt **RegexTokenizer**:
  - Sử dụng regex `\w+|[^\w\s]` để tách từ và dấu câu.
- Tạo `Lab1/main.py` để chạy thử trên ví dụ và dataset `UD_English-EWT`.

###  Học được
- Sự khác biệt giữa **tokenizer thủ công** và **tokenizer regex**.


---

## 📌 Lab 2: Vectorization
###  Nội dung thực hiện
- Xây dựng **interface `Vectorizer`** với 3 phương thức:
  - `fit(corpus)`
  - `transform(documents)`
  - `fit_transform(corpus)`
- Cài đặt **CountVectorizer**:
  - Nhận vào một `Tokenizer`.
  - Học **vocabulary** từ corpus.
  - Biến đổi văn bản thành **bag-of-words vector**.
- Tạo test (`test/lab2_test.py`) để chạy với ví dụ.

### Học được
- Cách cài đặt thủ công mô hình Bag-of-Words.
- Cách tích hợp tokenizer vào vectorizer.

---

## 📌 Lab 3: Trực quan hóa Word Embeddings với PCA và t-SNE
### Nội dung thực hiện
  - Sử dụng mô hình **GloVe pre-trained** (glove-wiki-gigaword-100) với 400K từ vựng.
  - Chọn **5,000 từ phổ biến nhất** để trực quan hóa.
  - Áp dụng **PCA (Principal Component Analysis)**:
      + Giảm chiều từ 100D xuống 2D bằng phương pháp tuyến tính.
      + Trực quan hóa phân bố từ trong không gian 2D.
  - Áp dụng **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
      + Giảm chiều phi tuyến với perplexity=30, max_iter=1000.
      + Tạo các cluster rõ ràng cho từ cùng ngữ nghĩa.
  - **So sánh PCA vs t-SNE**:
      + Vẽ biểu đồ song song để quan sát sự khác biệt.
      + Phân tích ưu/nhược điểm của từng phương pháp.
  - **Trực quan hóa nhóm từ theo chủ đề**
  - **Tìm và trực quan từ gần nghĩa**

### Học được
  - **Kỹ thuật giảm chiều** cho word embeddings:
      + PCA: Nhanh, ổn định, bảo toàn cấu trúc toàn cục.
      + t-SNE: Tạo cluster rõ ràng, bảo toàn cấu trúc cục bộ.
  - **So sánh PCA vs t-SNE**:
      + PCA: Tuyến tính → cluster không rõ ràng, từ phân tán đều.
      + t-SNE: Phi tuyến → cluster rõ ràng, từ cùng nghĩa nhóm gần nhau.
  - **Word embeddings học được mối quan hệ ngữ nghĩa**:
      + Từ cùng chủ đề (quốc gia, động vật, màu sắc) tập trung gần nhau.
      + Cosine similarity hiệu quả để tìm từ đồng nghĩa/gần nghĩa.


## 📌 Lab 4: Word Embedding
### Nội dung thực hiện
  1. lab4_test.py
  - Xây dựng lớp WordEmbedder để làm việc với word embeddings.
  - Cài mô hình pre-trained GloVe (glove-wiki-gigaword-50) bằng thư viện gensim.
  - Cài đặt các hàm xử lý:
      + get_vector(word): Lấy vector của một từ.
      + get_similarity(word1, word2): Tính độ tương đồng cosine giữa hai từ.
      + get_most_similar(word): Tìm các từ gần nghĩa nhất.
  - Hàm embed_document(document): Biểu diễn vector của một văn bản bằng cách lấy trung bình cộng các vector của các từ trong văn bản đó.
  - Tạo test (`test/lab4_test.py`) để chạy với ví dụ.

  2. lab4_embedding_trainning_demo.py
  - Xây dựng lớp StreamSentences để đọc dữ liệu lớn từ file theo từng đoạn (Tiết kiệm RAM), tự động tách câu/đoạn và tokenize.
  - Sử dụng thư viện gensim để huấn luyện mô hình Word2Vec từ trên corpus UD_English-EWT.
  - Tạo `test/lab4_embedding_trainning_demo.py` để chạy demo:
    + Tìm các từ có độ tương đồng cao (most_similar).
    + Giải bài toán quan hệ từ (Anology) (ví dụ: king - man + queen = ?).

  3. lab4_spark_word2vec_demo.py
  - Sử dụng Apache Spark và thư viện MLlib để xử lý dữ liệu lớn.
  - Khởi tạo SparkSession và làm việc với tập dữ liệu lớn.
  - Đọc và tiền xử lý dữ liệu văn bản bằng các phép biến đổi của Spark DataFrame:
    + Chuyển chữ thường (lower).
    + Loại bỏ ký tự đặc biệt (regexp_replace).
  - Sử dụng Tokenizer của Spark ML để tách từ.
  - Huấn luyện mô hình Word2Vec trên DataFrame đã xử lý bằng pyspark.ml.feature.Word2Vec.
  - Sử dụng phương thức findSynonyms của mô hình đã huấn luyện để tìm các từ tương đồng.
  - Tạo `test/lab4_spark_word2vec_demo.py` để thực thi toàn bộ pipeline xử lý dữ liệu và huấn luyện mô hình trên Spark.

### Học được
  1. lab4_test.py
    - Sử dụng các mô hình word embedding pre-trained như GloVe để biểu diễn vector của từ.
    - Tìm từ tương tự, tính độ tương đồng.
    - Tạo vector cho văn bản (document embedding) từ các word embedding có sẵn.
  2. lab4_embedding_trainning_demo.py
    - Cách huấn luyện một mô hình Word2Vec từ đầu bằng thư viện gensim.
    - Sự khác biệt giữa việc sử dụng mô hình pre-trained (`lab4_test.py`) và tự huấn luyện mô hình word embedding trên dữ liệu riêng.
    - Kỹ thuật xử lý dữ liệu lớn (corpus) để huấn luyện mô hình.
  3. lab4_spark_word2vec_demo.py
    - Cách xây dựng một pipeline xử lý ngôn ngữ tự nhiên với Apache Spark.
    - Sự khác biệt về tốc độ làm việc với gensim so với Spark MLlib trên dữ liệu quy mô lớn.
    - Cách sử dụng các thư viện của Spark ML như Tokenizer và Word2Vec.

---

## 📌 Lab 5: Text Classification - Sentiment Analysis
### Nội dung thực hiện
  1. **lab5_test.py** - TextClassifier với Scikit-learn
  - Xây dựng lớp **TextClassifier** (`src/models/text_classifier.py`) để phân loại văn bản.
  - Tích hợp với **TfidfVectorizer** đã xây dựng ở Lab 2.
  - Sử dụng **Logistic Regression** từ scikit-learn làm mô hình phân loại.
  - Cài đặt các phương thức:
      + `fit(texts, labels)`: Huấn luyện mô hình trên dữ liệu.
      + `predict(texts)`: Dự đoán nhãn cho văn bản mới.
      + `evaluate(y_true, y_pred)`: Đánh giá mô hình với các metrics (accuracy, precision, recall, f1-score).
  - Tạo test (`test/lab5_test.py`) với dữ liệu mẫu 6 câu phân loại.

  2. **lab5_spark_sentiment_analysis.py** - Baseline với PySpark
  - Xây dựng pipeline phân tích cảm xúc baseline với **PySpark MLlib**.
  - Khởi tạo **SparkSession** và đọc dữ liệu từ `sentiments.csv`.
  - Tiền xử lý dữ liệu:
      + Chuyển đổi nhãn từ {-1, 1} thành {0, 1}.
      + Loại bỏ giá trị null.
  - Xây dựng Pipeline gồm:
      + **Tokenizer**: Tách văn bản thành từ.
      + **StopWordsRemover**: Loại bỏ stop words.
      + **HashingTF** (10,000 features): Chuyển từ thành vector TF.
      + **IDF**: Tính trọng số IDF.
      + **LogisticRegression**: Mô hình phân loại với maxIter=10, regParam=0.001.
  - Đánh giá mô hình trên test set với accuracy, precision, recall, f1-score.
  - **Kết quả**: Accuracy 72.25%.

  3. **lab5_improvement_test.py** - So sánh và cải tiến mô hình
  - Cải tiến pipeline với **tiền xử lý văn bản nâng cao**:
      + Chuyển chữ thường, loại bỏ URL, HTML tags.
      + Loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng.
  - So sánh **3 mô hình**:
      + **Baseline**: HashingTF (10,000) + IDF + Logistic Regression → 72.25%
      + **Improved**: Tiền xử lý + HashingTF (2,000) + IDF + **GBTClassifier** (100 iterations) → 76.29%
      + **Neural Network**: Tiền xử lý + HashingTF (5,000) + IDF + **MLP [5000,64,32,2]** (150 iterations) → **76.46%** 

### Học được
  1. **lab5_test.py**
    - Cách xây dựng pipeline phân loại văn bản hoàn chỉnh.
    - Tích hợp tokenizer, vectorizer vào classifier.
    - Sử dụng các metrics đánh giá mô hình phân loại.
  
  2. **lab5_spark_sentiment_analysis.py**
    - Xây dựng pipeline Machine Learning với PySpark MLlib.
    - Xử lý dữ liệu văn bản quy với Spark DataFrame.
  
  3. **lab5_improvement_test.py**
    - Tầm quan trọng của **tiền xử lý văn bản** và **số lượng features** (HashingTF) trong việc cải thiện accuracy.
    - So sánh hiệu suất các mô hình: Logistic Regression vs GBTClassifier vs Neural Network.
    - Trade-off giữa **độ chính xác** và **thời gian huấn luyện**.

## 📌 Lab 5.1: Deep Learning for NLP

### Lab 5 Part 1: Introduction to PyTorch
#### Nội dung thực hiện
  - **Task 1: Tensor Operations**
      + Tạo tensor từ list, NumPy array, random values.
      + Các phép toán: cộng, nhân, matrix multiplication.
      + Indexing, slicing và reshape tensor.
      + Thuộc tính: shape, dtype, device.
  - **Task 2: Autograd**
      + Tính đạo hàm tự động với `requires_grad=True`.
      + Sử dụng `.backward()` để tính gradient.
      + Gradient accumulation và `retain_graph=True`.
  - **Task 3: Neural Network Modules**
      + **nn.Linear**: Fully connected layer (5 dims → 2 dims).
      + **nn.Embedding**: Chuyển word indices thành dense vectors (10 từ → 3 dims).
      + **nn.Module**: Xây dựng mô hình tùy chỉnh với Embedding → Linear → ReLU → Output.

#### Học được
  - Tensor operations là nền tảng cho Deep Learning.
  - Autograd giúp tính gradient tự động cho backpropagation.
  - Xây dựng neural networks với torch.nn modules.
  - Kiến trúc cơ bản: Embedding → Hidden layers → Output.

---

### Lab 5 Part 2: RNNs for Text Classification
#### Nội dung thực hiện
  - **Dataset**: HWU64 Intent Classification (64 lớp intent, 8954 train / 1076 val / 1076 test).
  - **Task 1: Baseline - TF-IDF + Logistic Regression**
      + TfidfVectorizer (5000 features) + LogisticRegression.

  - **Task 2: Word2Vec + Dense Neural Network**
      + Huấn luyện Word2Vec (vector_size=200, sg=1, epochs=20).
      + Chuyển câu thành vector trung bình.
      + Dense network: 200 → 256 → 128 → 64 classes.

  - **Task 3: Pre-trained Embedding + Bi-LSTM**
      + Tokenize + Padding (max_len=50).
      + Embedding Matrix từ Word2Vec (trainable=False).
      + Bidirectional LSTM (128 + 64 units).

  - **Task 4: Embedding from Scratch + Bi-LSTM**
      + Embedding learnable (200 dims, trainable=True).
      + Bidirectional LSTM với same architecture.

  - **Task 5: So sánh và đánh giá**
      + Định lượng: F1-score macro trên test set.
      + Định tính: Test trên 20 câu.

#### Học được
  - **RNN/LSTM** bắt được thứ tự từ, tốt hơn lấy trung bình của chuỗi rồi phân loại.
  - **Embedding from scratch** tốt hơn pre-trained khi có đủ data và task chuyên biệt.
  - **Bidirectional LSTM** mạnh mẽ với text classification (context 2 chiều).
  - Các kỹ thuật: Dropout, BatchNorm, EarlyStopping, ReduceLROnPlateau.

---

### Lab 5 Part 3: RNNs for POS Tagging
#### Nội dung thực hiện
  - **Dataset**: UD English-EWT (Universal Dependencies) - 17 UPOS tags.
  - **Task 1: Chuẩn bị dữ liệu**
      + Load CoNLL-U format (train: 12,543 câu / dev: 2,001 / test: 2,077).
      + Xây dựng vocabulary: 19,675 từ, 17 POS tags.
  - **Task 2: PyTorch Dataset và DataLoader**
      + Custom POSDataset class.
      + Padding với `pad_sequence` và `collate_fn`.
      + DataLoader với batch_size=32.
  - **Task 3: Xây dựng mô hình SimpleRNN**
      + Embedding (19675 → 128 dims).
      + RNN (128 → 256 hidden).
      + Linear (256 → 17 tags).
  - **Task 4: Huấn luyện**
      + Optimizer: Adam (lr=0.001).
      + Loss: CrossEntropyLoss với ignore_index=PAD_TAG_ID.
      + Training 5 epochs với masking.
  - **Task 5: Đánh giá**
      + **Dev Accuracy**: 88.46%.
      + **Test Accuracy**: 88.15%.
      + Dự đoán câu mới: "The quick brown fox jumps over the lazy dog".

#### Học được
  - **Sequence labeling** khác text classification (mỗi token có 1 label).
  - **Padding và Masking** quan trọng để xử lý câu khác độ dài.
  - **RNN** phù hợp cho POS tagging (bắt được temporal dependencies).

---

### Lab 5 Part 4: RNNs for Named Entity Recognition (NER)
#### Nội dung thực hiện
  - **Dataset**: CoNLL-2003 NER (14,041 train / 3,250 validation / 3,453 test).
  - **Nhãn**: 9 NER tags (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).
  - **Task 1: Chuẩn bị dữ liệu**
      + Load dataset từ Hugging Face datasets.
      + Chuyển đổi nhãn số sang nhãn string.
      + Xây dựng vocabulary: 23,624 từ, 10 tags (9 NER + PAD).
  - **Task 2: PyTorch Dataset và DataLoader**
      + Custom NERDataset class.
      + Padding với attention mask.
      + DataLoader batch_size=32.
  - **Task 3: Xây dựng Bidirectional LSTM**
      + Embedding (23624 → 128 dims).
      + Bidirectional LSTM (128 → 256×2 hidden, dropout=0.5).
      + Linear (512 → 10 tags).
  - **Task 4: Huấn luyện 5 epochs**
      + Loss giảm: 0.4993 → 0.0208.
      + Train accuracy: 92.45% → 99.90%.
      + Valid accuracy: 90.90% → 94.69%.
  - **Task 5: Đánh giá**
      + **Validation**: Accuracy 94.69%, Precision 76.75%, Recall 70.23%, F1 73.35%.
      + **Test**: Accuracy 92.67%, Precision 68.51%, Recall 61.60%, F1 64.87%.
      + Dự đoán các câu mới.

#### Học được
  - **Bidirectional LSTM** hiệu quả cho NER (context 2 chiều).
  - Sử dụng **seqeval** cho entity-level metrics.
  - NER phức tạp hơn POS tagging vì cần nhận dạng cả ranh giới thực thể.

---

## 📌 Lab 6: Transformers và Hugging Face
### Nội dung thực hiện
  - **Bài 1: Masked Language Modeling (MLM)**
      + Sử dụng pipeline "fill-mask" với BERT.
      + Test câu: "Hanoi is the `<mask>` of Vietnam".
      + **Kết quả**: Dự đoán "capital" với độ tin cậy 40.33%.
  
  - **Bài 2: Text Generation**
      + Sử dụng pipeline "text-generation" với GPT-2.
      + Prompt: "The best thing about learning NLP is".
      + **Kết quả**: Văn bản mạch lạc nhưng chưa trả lời đúng trọng tâm.

  - **Bài 3: Sentence Embedding**
      + Load mô hình BERT (bert-base-uncased, 768 hidden dims).
      + Tokenize câu: "This is a sample sentence.".
      + Lấy hidden states từ last layer.
      + **Mean Pooling** với attention_mask.
      + **Kết quả**: Vector 768 chiều.

### Học được
  - **Ba kiến trúc Transformer**:
      + Encoder-only (BERT): Hiểu ngữ cảnh → Classification, NER, QA.
      + Decoder-only (GPT): Sinh văn bản → Text generation.
      + Encoder-Decoder (T5): Seq2seq → Translation, Summarization.
  - **Hugging Face Transformers**:
      + Pipeline API đơn giản hóa việc sử dụng mô hình.
  - **Masked Language Modeling**: BERT dự đoán từ bị mask bằng bidirectional context.
  - **Text Generation**: GPT sinh văn bản.
  - **Sentence Embedding**: Mean pooling với attention mask tạo vector biểu diễn câu.

---

## 📌 Lab 7: Dependency Parsing
### Nội dung thực hiện
  - **Phần 1: Cài đặt**
      + Cài đặt thư viện **spaCy**.
      + Tải mô hình ngôn ngữ tiếng Anh `en_core_web_sm`.
  
  - **Phần 2: Phân tích và trực quan hóa**
      + Load mô hình spaCy và phân tích câu.
      + Sử dụng **displaCy** để trực quan hóa cây phụ thuộc.
      + Khởi chạy server tại http://127.0.0.1:5000 để xem cây phụ thuộc.
  
  - **Phần 3: Truy cập các thành phần trong cây**
      + Truy cập các thuộc tính token: `text`, `pos_`, `dep_`, `head`, `children`.
      + Phân tích câu "Apple is looking at buying U.K. startup for $1 billion".
      + Hiểu quan hệ giữa token, head và children trong cây phụ thuộc.
  
  - **Phần 4: Duyệt cây để trích xuất thông tin**
      + **Bài toán 4.1**: Tìm chủ ngữ (nsubj) và tân ngữ (dobj) của động từ.
        - Test: "The cat chased the mouse and the dog watched them."
        - Kết quả: (cat, chased, mouse) và (dog, watched, them).
      + **Bài toán 4.2**: Tìm tính từ bổ nghĩa (amod) cho danh từ.
        - Test: "The big, fluffy white cat is sleeping on the warm mat."
        - Kết quả: cat → [big, fluffy, white], mat → [warm].
  
  - **Phần 5: Bài tập tự luyện**
      + **Bài 1**: Tìm động từ chính (ROOT) của câu.
        - Viết hàm `find_main_verb(doc)` trả về token có `dep_ == "ROOT"`.
        - Test 4 câu và hiển thị động từ chính, POS tag, lemma.
      
      + **Bài 2**: Trích xuất các cụm danh từ (Noun Chunks).
        - Viết hàm `extract_noun_chunks(doc)` tự động trích xuất cụm danh từ.
        - Thu thập danh từ và các từ bổ nghĩa: det, amod, compound, nummod, poss.
        - So sánh kết quả với `.noun_chunks` có sẵn của spaCy.
      
      + **Bài 3**: Tìm đường đi ngắn nhất trong cây.
        - Viết hàm `get_path_to_root(token)` tìm đường đi từ token lên ROOT.
        - Viết hàm `get_distance_to_root(token)` tính số bước đến ROOT.

### Học được
  - **Dependency Parsing**: Hiểu cấu trúc cây phụ thuộc và các quan hệ giữa từ.
  - **spaCy**: Sử dụng thư viện mạnh mẽ cho phân tích cú pháp phụ thuộc.
  - **displaCy**: Trực quan hóa cây phụ thuộc giúp hiểu rõ cấu trúc câu.
  - **Trích xuất thông tin**: Tìm Subject-Verb-Object, tính từ bổ nghĩa, noun chunks.
  - **Thuật toán trên cây**: Duyệt cây, tìm đường đi, tính khoảng cách, LCA.
  - **Ứng dụng thực tế**: Information Extraction, Question Answering, Knowledge Graph Construction.

---

## 📌 Lab X: Text-to-Speech (Nghiên cứu)
### Nội dung nghiên cứu
  - **Lịch sử phát triển TTS**:
      + Giai đoạn truyền thống: Concatenative, Formant, Articulatory Synthesis.
      + Kỷ nguyên Deep Learning: WaveNet (2016), Tacotron (2017).
      + Thế hệ hiện đại: Tacotron 2, FastSpeech, VITS, Neural Codec Models.
  
  - **Kiến trúc TTS hiện đại**:
      + Pipeline: Text Analysis → Acoustic Model → Vocoder → Audio.
      + **Tacotron 2**: Character Embeddings → Encoder → Attention → Decoder → Post-net.
      + **FastSpeech 2**: Feed-Forward Transformer + Variance Adaptors (duration, pitch, energy).
      + **HiFi-GAN**: GAN-based vocoder với Multi-receptive field fusion.
  
  - **Kỹ thuật nâng cao**:
      + Multi-Speaker TTS: Speaker embeddings, Zero-shot voice cloning.
      + TTS biểu cảm: Style Tokens, Reference Encoder, emotion control.
      + Cross-lingual TTS: IPA phonemes, language embeddings, transfer learning.
  
  - **Thách thức & Tương lai**:
      + Ngữ điệu tự nhiên, zero-shot cloning, mô hình thống nhất.
      + Hiệu năng on-device, phát hiện Deepfake, watermarking.

---

