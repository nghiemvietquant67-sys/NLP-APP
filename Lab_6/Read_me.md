# Lab 6: Giới thiệu về Transformers


#### Bài 1 – Fill-Mask (Masked Language Modeling)
**Mô hình tự động tải:** `vinai/phobert-base-v2` (multilingual)  
**Câu đầu vào:** `Hanoi is the [MASK] of Vietnam.`

**Top 5 dự đoán thực tế:**
1. **capital**    → 0.9341 (đúng, xếp thứ 1)  
2. Republic      → 0.0300  
3. Capital       → 0.0105  
4. birthplace    → 0.0054  
5. heart         → 0.0014  

**Trả lời câu hỏi:**  
1. Có, mô hình dự đoán đúng từ “capital” ở vị trí đầu tiên.  
2. BERT (Encoder-only + bidirectional) rất phù hợp vì có thể nhìn toàn bộ ngữ cảnh hai chiều xung quanh token [MASK].

#### Bài 2 – Text Generation (Next Token Prediction)
**Mô hình:** GPT-2 (mặc định)  
**Prompt:** `The best thing about learning NLP is`  
**Tham số:** `max_length=60`, `num_return_sequences=3` (các warning là bình thường)

**Một kết quả tiêu biểu (mạch lạc, tự nhiên):**
The best thing about learning NLP is that you don't need to understand every aspect of it. You can learn it as you go along...
text**Trả lời câu hỏi:**  
1. Kết quả hoàn toàn hợp lý, đúng ngữ pháp và phù hợp ngữ cảnh.  
2. GPT-2 (Decoder-only) được huấn luyện với mục tiêu next-token prediction + causal attention → lý tưởng cho tác vụ sinh văn bản.

#### Bài 3 – Sentence Embedding bằng Mean Pooling
**Mô hình:** `bert-base-uncased` (đúng chuẩn yêu cầu bài tập)  
**Câu đầu vào:** `This is a sample sentence.`

**Kết quả thực tế:**
Vector biểu diễn (một phần đầu):
tensor([-6.3874e-02, -4.2837e-01, -6.6779e-02, -3.8430e-01, -6.5784e-02,
-2.1826e-01,  4.7636e-01,  4.8659e-01,  4.0647e-05, -7.4273e-02])
Kích thước vector: torch.Size([1, 768])
text**Trả lời câu hỏi:**  
1. Kích thước vector là **768** → chính là tham số **hidden_size** của BERT-base.  
2. Dùng `attention_mask` để loại bỏ ảnh hưởng của các token padding ([PAD]), tránh làm sai lệch vector biểu diễn của câu thực tế.
