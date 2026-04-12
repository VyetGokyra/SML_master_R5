# Báo cáo Benchmark các Kỹ thuật Prompt trên Dataset COPA (Commonsense Reasoning)

**COPA (Choice Of Plausible Alternatives)** là bộ Benchmark đo lường "Kiến thức Thường thức" của trí tuệ nhân tạo thông qua việc đánh giá nguyên nhân/hệ quả. VD: "Người phụ nữ đi tắm. Tại sao? 1. Cô ấy vừa tập gym. 2. Cô ấy đang đói."
Khác với RTE (tuyần túy logic) và FinQA (thuần túy toán), COPA đòi hỏi LLM phải nắm được quy luật đời sống ngoài đời thực. Đây là sân khấu biểu diễn đỉnh cao của các mô hình Causal LM (Decoder-only) với khối lượng dữ liệu Pre-training khổng lồ.

---

## 1. Kết quả trên Kiến trúc BERT (`bert-base-uncased`)
*Vì BERT suy diễn logic dựa trên các từ đồng xuất hiện (co-occurance), nó gặp rào cản lớn với thế giới thường thức.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Cloze + RAG** | **57.00%** | Mức cao nhất của BERT, được cải thiện một chút nhờ 1 shot RAG mồi ngữ cảnh. |
| **Prefix Zero-shot** | 55.00% | Chỉ nhỉnh hơn đoán bừa (50%) một chút. |
| **Decompose + RAG** | 55.00% | Bóc nhỏ khía cạnh câu hỏi nhưng BERT không đủ sức ráp lại. |
| **Prefix + RAG** | 53.00% | Gần như không hiểu RAG. |
| **Prefix + CoT** | 52.00% | Tương tự như đoán cầu may. |
| **PET-Style** | 49.00% | Thua cả đoán bừa 50%. Cách ghép câu của PET không hợp với tiếng Anh thường thức. |
| **Cloze Simple** | 46.00% | Dứt mạch logic ở `[MASK]` khiến mô hình hụt hẫng. |

---

## 2. Kết quả trên Kiến trúc Flan-T5 (`google/flan-t5-base`)
*Dù được Instruction-tuned kỹ càng, Base Model của Flan vẫn gặp lỗi xuất định dạng "lệch tủ".*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Cloze + RAG** | **62.00%** | Nhờ điền khuyết RAG, Flan-T5 biết mình phải sinh đáp án "1" hoặc "2". |
| **PET-Style** | 55.00% | Format chuẩn mực giúp Flan hiểu đề. |
| **Prefix + RAG** | 54.00% | In-context có hỗ trợ vừa đủ. |
| **Prefix + CoT** | 23.00% | Sinh "ảo giác" nặng nề và lạc đề hoàn toàn khi giải thích dông dài. |
| **Prefix Zero-shot** | 17.00% | Không có RAG mồi tủ, mô hình thường tự chép lại toàn bộ câu trả lời thay vì nhả đáp án 1 hoặc 2 làm rớt điểm EM. |
| **Decompose + RAG** | 4.00% | Việc chia tách `cause/effect` quá sức và không tương thích với Seq2Seq. |
| **Cloze Simple** | 2.00% | Rớt sạch điểm vì Model không tự phác thảo được hình dáng Output. |

---

## 3. Kết quả trên Kiến trúc Qwen3 (`Qwen3-0.6B`)
*Nhờ kiến thức đời sống trong Pre-training, Causal LM bứt tốc cực thét.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Decompose + RAG** | **94.00%** | Sức mạnh kinh hoàng! Việc tự bóc tách Main Subject rồi phán đoán Cause/Effect nâng tầm lý luận lên sát 100%. |
| **Cloze Simple** | 69.00% | Cho Causal tự điền tiếp câu ngầm định luôn đạt hiệu quả cao với Common Sense. |
| **Prefix Zero-shot** | 64.00% | Nhảy vọt so với BERT, hiểu rõ ngữ cảnh đời sống. |
| **Prefix + RAG** | 59.00% | Kém tinh tế hơn Zero-shot vì bị chi phối bởi các văn cảnh lạ của Few-shot. |
| **Cloze + RAG** | 49.00% | RAG kết hợp Cloze phá vỡ mạch logic tự nhiên. |
| **Prefix + CoT** | 35.00% | Bản thân 0.6B không gánh được việc múa chữ lằng nhằng. |
| **PET-Style** | 11.00% | Không thiết kế cho luồng sinh một chiều. |

---

## 4. Kết quả trên Kiến trúc Qwen3.5 (`Qwen3.5-0.8B`)
*Phiên bản lớn hơn tỏa sáng theo một logic "đậm chất" con người.*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Prefix + CoT** | **97.00%** | Tuyệt đỉnh Chain-of-Thought! Với Common Sense, khi yêu cầu nghĩ Step-by-Step, mô hình khơi dậy được luồng học đời sống và suy ra nguyên nhân chuẩn xác phi thường! |
| **Prefix Zero-shot** | 67.00% | Base model đáp lời cực gắt, không cần mồi nhử. |
| **Decompose + RAG** | 60.00% | Tạm ổn, nhưng không bằng CoT. |
| **Cloze Simple** | 59.00% | Tự động sinh sinh đáp án tốt nhờ khả năng Next-token. |
| **Prefix + RAG** | 58.00% | Vẫn bị hội chứng "nhiễu Few-shot" hệt bản nhỏ. |
| **Cloze + RAG** | 52.00% | Loay hoay giữa RAG và dòng sinh chữ. |
| **PET-Style** | 37.00% | Cấu trúc PET hoàn toàn dị biệt với Qwen. |

---
## 💡 Kết luận Chung (Insight Học Thuật Đại Diện Benchmark Thứ 3)

**Thứ nhất:** Khác với những tác vụ đọc hiểu nhàm chán, Commonsense (Thường thức) khẳng định vị trí độc tôn của **Causal LM (Decoder-only)**. Qwen dễ dàng chạm mốc `94-97%` bởi lượng trí tuệ ngầm ẩn (Latent Knowledge) từ dữ liệu Pre-training lớn vượt bậc so với BERT. Quá trình sinh một chiều hoạt động như não bộ người đang "hồi tưởng" lại đời sống tự nhiên.

**Thứ hai:** Chain-of-Thought (CoT) từng là "tội đồ" trong tập FinQA do ảo giác toán học, từng là "kẻ ngáng đường" trong NLI... thì tại bài toán `COPA`, **CoT trở thành Anh Hùng (Đạt 97%)**. Điều này chứng minh: *Không có Kỹ thuật Prompt nào là vạn năng, sức mạnh của Prompt + CoT chỉ khuếch đại tối đa khi dấn thân vào các chủ đề xã hội / đời sống mở.* Cực kỳ đắt giá!
