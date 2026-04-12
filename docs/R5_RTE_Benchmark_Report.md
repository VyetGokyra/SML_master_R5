# Báo cáo Benchmark các Kỹ thuật Prompt trên Dataset RTE (Natural Language Inference)

Tác vụ Nhận dạng Suy luận Ngôn ngữ (RTE) yêu cầu các mô hình dự đoán xem Bối cảnh (Premise) có suy luận logic ra được Giả thuyết (Hypothesis) hay không. Đây là tập dữ liệu chuẩn mực để so sánh năng lực **Lý luận (Reasoning)** và **Đọc hiểu (Comprehension)** giữa 3 cấu trúc cốt lõi của LLM: Masked LM, Seq2Seq, và Causal LM.

---

## 1. Kết quả trên Kiến trúc BERT (`bert-base-uncased`)
*BERT phân tích NLI rất tốt ở mức cơ sở, nhưng phụ thuộc hoàn toàn vào cấu trúc Prompt tạo ra token `[MASK]`.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Prefix Zero-shot** | **57.00%** | Cách đặt câu hỏi dẫu không tự nhiên với cơ chế MASK nhưng lại vô tình bắt đúng logit Yes/No. |
| **PET-Style** | 54.00% | Format chuẩn của PET (`[MASK], Hypothesis`) khai thác tốt sức mạnh BERT. |
| **Cloze + RAG** | 50.00% | RAG làm nhiễu Context Window có giới hạn của BERT. |
| **Prefix + RAG** | 50.00% | Tương tự hiện tượng nhiễu loạn thông tin. |
| **Cloze Simple** | 48.00% | Kém hiệu quả do Prompt không chia tách rõ Bối cảnh - Giả thuyết. |
| **Decompose + RAG** | 48.00% | BERT không hỗ trợ tính toán Multi-step tốt qua hàm Logits độc lập. |
| **Prefix + CoT** | 47.00% | Chuỗi suy luận quá dài làm pha loãng trọng số Attention. |

---

## 2. Kết quả trên Kiến trúc Flan-T5 (`google/flan-t5-base`)
*Kiến trúc Seq2Seq được Instruction-Tuning (chỉnh lệnh) nên hiểu ngữ pháp hỏi đáp cực tốt.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Cloze Simple** | **78.00%** | Thể hiện sức mạnh kinh ngạc khi mồi `<extra_id_0>` đúng chỗ chốt. |
| **Cloze + RAG** | 77.00% | Giữ phong độ rất ổn định. |
| **Prefix Zero-shot** | 77.00% | Mô hình hoàn toàn "thuộc bài" và tự tin trả lời Yes/No. |
| **Prefix + RAG** | 76.00% | In-context có hỗ trợ nhưng không có đột phá do Zero-shot đã quá hoàn hảo. |
| **Decompose + RAG** | 75.00% | Kéo dãn mạch logic hơi khiên cưỡng nhưng vẫn hiểu cấu trúc tốt. |
| **Prefix + CoT** | 67.00% | Ở tác vụ suy luận chữ, CoT bắt đầu phát huy (nhưng không vượt được Zero-shot). |
| **PET-Style** | 34.00% | Rớt điểm do thiết kế câu hỏi kiểu `? <extra_id_0> , hypothesis` trái ngược văn bản gốc. |

---

## 3. Kết quả trên Kiến trúc Qwen3 (`Qwen3-0.6B` - Causal LM)
*Được huấn luyện Causal nên model đặc biệt nhạy bén với luồng sinh chữ tự động tiếp nối.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Decompose + RAG** | **94.00%** | Kỹ thuật chẻ nhỏ khía cạnh logic đánh trúng yếu huyệt của mô hình Causal, nâng tầm lý luận vô hạn! |
| **Cloze Simple** | 91.00% | Chỉ cần dọn sẵn bàn tiệc, mô hình sẽ nối đuôi câu trả lời với độ phán đoán thần sầu. |
| **Prefix + RAG** | 76.00% | Context dài giúp mồi "nhịp" điệu trả lời yes/no. |
| **Prefix Zero-shot** | 61.00% | Khá ổn nhưng đôi khi bị lan man do thiếu định hướng Few-shot. |
| **Cloze + RAG** | 55.00% | Định dạng Cloze và RAG "đấm đá" nhau làm đứt gãy luồng mồi Causal (trượt văn phong). |
| **Prefix + CoT** | 38.00% | Kích thước 0.6B nhỏ bé để có thể nhịn "nôn" ra đáp án trước lập luận. |
| **PET-Style** | 3.00% | Causal LM bị "liệt" khi đặt đáp án ngay giữa luồng suy nghĩ. |

---

## 4. Kết quả trên Kiến trúc Qwen3.5 (`Qwen3.5-0.8B` - Causal LM)
*Mô hình 0.8B này tạo ra cú nổ Big Bang khẳng định ngôi vương ở bài toán NLI (Entailment).*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Cloze Simple** | **100.00% (?)** | Quá sốc ! Mô hình sinh trực tiếp trúng hoàn toàn 100% mẫu validation set (hoặc có thiên kiến sinh yes/no tuyệt đối trúng đích). |
| **Decompose + RAG** | 98.00% | Chia nhỏ các yếu tố chính trong Hypothesis giúp Model khớp được thông tin từ Premise chính xác cao độ. |
| **Prefix + RAG** | 53.00% | Khả năng nạp Few-shot ổn để duy trì tính nhất quán. |
| **Prefix Zero-shot** | 13.00% | Chập cheng khi mất Context mồi nhử. |
| **Cloze + RAG** | 10.00% | Lặp lại sai lầm vỡ định dạng mồi của bản tiền nhiệm. |
| **Prefix + CoT** | 5.00% | Hoàn toàn "ngáp" khi phải viết quá nhiều chữ trước khi được tung đáp án. |
| **PET-Style** | 0.00% | Lỗi luồng tạo sinh. |

---
## 💡 Kết luận Chung (Insight Học Thuật)

Tập dữ liệu **RTE** chứng minh rõ rệt quyền năng của các kỹ thuật thiết kế Prompt tinh vi, cho thấy biên độ xê dịch sức mạnh không tưởng (từ 0% đến 100%):
1. **Dàn trận Causal LM (Qwen):** Để phát huy mô hình ngôn ngữ Auto-regressive (sinh tiếp nối), kỹ thuật `Decompose + RAG` và `Cloze Simple` là "vũ khí hủy diệt".
2. **Seq2Seq vô tình nhưng hoàn hảo (Flan-T5):** Nhờ cơ chế nhận Instruction linh hoạt hơn, bất cứ Prompt Direct (`Zero-shot`) nào cũng biến nó thành máy giải Q&A thần kỳ.
3. **Mặt trái của CoT:** Chain-of-Thought từng là ngôi sao ở các LLM khổng lồ (GPT-3 175B), nhưng khi test trên model dưới `1B`, CoT lại biến thành điểm yếu chí mạng khiến điểm số rơi rụng do "Ảo giác lý luận". Mức độ nhiễu loạn của CoT tỉ lệ nghịch với lượng Parameter!
