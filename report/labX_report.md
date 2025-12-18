# Báo cáo TTS (Text-to-Speech)

## Tổng quan bài toán & tình hình nghiên cứu

**Text-to-Speech (TTS)** là công nghệ chuyển đổi văn bản thành giọng nói tự nhiên. Quá trình nghiên cứu và phát triển TTS đã trải qua nhiều giai đoạn, từ các phương pháp dựa trên quy tắc (rule-based) vào thập niên 1980, sang các mô hình **deep learning** từ khoảng năm 2016, và hiện nay là các mô hình **few-shot kết hợp LLM** (giai đoạn 2023–2025) như *VALL-E* hay *Chatterbox*.

Xu hướng nghiên cứu hiện tại tập trung vào các mục tiêu chính:

* Giọng nói có **biểu cảm và cảm xúc tự nhiên** hơn.
* Khả năng **đa ngôn ngữ**, mở rộng tới hơn 1000 ngôn ngữ.
* **Thời gian suy luận nhanh (realtime)** để ứng dụng trong thực tế.
* Các vấn đề **đạo đức và an toàn**, như nhúng watermark để chống giả mạo giọng nói (deepfake).

---

## Các phương pháp triển khai chính

### Level 1: Rule-based (Formant Synthesis)

**Ưu điểm**:

* Tốc độ xử lý nhanh, tiêu thụ ít tài nguyên.
* Hỗ trợ đa ngôn ngữ tốt.
* Dễ kiểm soát và triển khai.

**Nhược điểm**:

* Giọng nói mang tính máy móc, thiếu tự nhiên.
* Hạn chế về cảm xúc và ngữ điệu.

**Phù hợp**:

* Thiết bị nhúng và IoT.
* Ứng dụng cho các ngôn ngữ ít dữ liệu.
* Ví dụ: hệ thống đọc thông báo trên đồng hồ thông minh.

---

### Level 2: Deep Learning (Tacotron 2, FastSpeech, WaveNet)

**Ưu điểm**:

* Giọng nói tự nhiên, gần với giọng người.
* Có khả năng biểu đạt cảm xúc.
* Dễ cá nhân hóa thông qua fine-tuning.

**Nhược điểm**:

* Yêu cầu lượng dữ liệu huấn luyện lớn.
* Tốn tài nguyên tính toán.
* Gặp khó khăn với các ngôn ngữ thiểu số.

**Phù hợp**:

* Audiobook.
* Trợ lý ảo (ví dụ: Google Assistant).
* Ứng dụng e-learning, đặc biệt cho tiếng Việt.

---

### Level 3: Few-shot (VALL-E, Chatterbox)

**Ưu điểm**:

* Có thể sao chép giọng nói chỉ với 3–5 giây mẫu âm thanh.
* Biểu cảm cao, linh hoạt.
* Khả năng đa ngôn ngữ mạnh.

**Nhược điểm**:

* Mô hình lớn, yêu cầu GPU mạnh.
* Chi phí tính toán cao.
* Tiềm ẩn rủi ro đạo đức liên quan đến deepfake.

**Phù hợp**:

* Voice cloning hỗ trợ người khuyết tật.
* Ứng dụng trong game.
* Sản xuất nội dung sáng tạo cá nhân.

---

## Pipeline tối ưu hóa (giảm nhược điểm, tăng ưu điểm)

### Đối với Level 1

* Kết hợp với từ điển phát âm tự động.
* Cải thiện độ tự nhiên của giọng nói trong khi vẫn duy trì tốc độ xử lý cao.

### Đối với Level 2

* Áp dụng **transfer learning** từ tiếng Anh sang tiếng Việt, chỉ cần khoảng 1–2 giờ dữ liệu.
* Sử dụng **knowledge distillation** để giảm kích thước mô hình, phù hợp cho thiết bị di động.
* Tích hợp **global style tokens** nhằm kiểm soát cảm xúc dễ dàng hơn.

### Đối với Level 3

* Áp dụng mô hình **hybrid**: few-shot chỉ dùng cho đặc trưng giọng (timbre), còn ngữ điệu (prosody) do mô hình nhỏ hơn đảm nhiệm.
* Nhúng watermark tự động và yêu cầu sự đồng thuận (consent) của người dùng để giảm rủi ro đạo đức.
* Ưu tiên **on-device inference** nhằm tăng tính bảo mật và quyền riêng tư.
