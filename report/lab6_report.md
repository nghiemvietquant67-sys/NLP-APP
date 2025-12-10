from pypandoc import convert_text

content = """# Báo cáo TTS (Text-to-Speech)

## Tổng quan bài toán & tình hình nghiên cứu
Text-to-Speech (TTS) là công nghệ chuyển văn bản thành giọng nói tự nhiên. Nghiên cứu phát triển từ phương pháp dựa trên quy tắc (thập niên 80) đến deep learning (2016+) và hiện nay là few-shot với LLM (2023-2025, ví dụ: VALL-E, Chatterbox). Xu hướng hiện tại tập trung vào giọng nói biểu cảm, đa ngôn ngữ (>1000 ngôn ngữ), tốc độ realtime và đạo đức (nhúng watermark chống deepfake).

## Các phương pháp triển khai chính

### Level 1: Rule-based (Formant Synthesis)
**Ưu điểm:** Nhanh, ít tài nguyên, hỗ trợ tốt đa ngôn ngữ, dễ kiểm soát.  
**Nhược điểm:** Giọng robot, thiếu tự nhiên và cảm xúc.  
**Phù hợp:** Thiết bị nhúng, IoT, ứng dụng ngôn ngữ ít dữ liệu.

### Level 2: Deep Learning (Tacotron 2, FastSpeech, WaveNet)
**Ưu điểm:** Giọng nói tự nhiên, cảm xúc, dễ fine-tune.  
**Nhược điểm:** Cần nhiều dữ liệu, tốn tài nguyên.  
**Phù hợp:** Audiobook, trợ lý ảo, e-learning.

### Level 3: Few-shot (VALL-E, Chatterbox)
**Ưu điểm:** Clone giọng 3–5 giây, đa ngôn ngữ.  
**Nhược điểm:** Model lớn, rủi ro deepfake.  
**Phù hợp:** Voice cloning cho người khuyết tật, game, sáng tạo nội dung.

## Pipeline tối ưu hóa
- **Level 1:** Thêm từ điển phát âm để tăng tự nhiên.  
- **Level 2:** Transfer learning, distillation giảm kích thước model, thêm style tokens.  
- **Level 3:** Kết hợp với Level 2, nhúng watermark, on-device inference.
"""

# Convert to markdown file (md)
output_path = "/mnt/data/tts_report.md"
convert_text(content, 'md', format='md', outputfile=output_path, extra_args=['--standalone'])

output_path
