# Báo cáo Benchmark các Kỹ thuật Prompt trên Dataset FinQA (Financial QA)

Tài liệu này cung cấp cái nhìn thực tế và tàn khốc nhất khi áp dụng **Prompt-based Learning (Zero-shot & Few-shot RAG)** lên một Dataset "cực kỳ khó nhằn" là **FinQA**. 

Tác vụ của FinQA yêu cầu khả năng: (1) tìm kiếm dữ liệu trên lưới Table và Text, (2) trích xuất số liệu, (3) tính toán logic/toán học. Kết quả dưới đây đã chứng minh rõ lý thuyết: Các mô hình Base LM dung lượng nhỏ (dưới 1 tỷ tham số) **thất bại gần như hoàn toàn** trước toán học tài chính nếu không có sự hỗ trợ của mô hình tính toán bên ngoài.

---

## 1. Kết quả trên Kiến trúc BERT (`bert-base-uncased`)
*Vì BERT chỉ dự đoán được duy nhất 1 Token (Word) cho dấu `[MASK]`, nó không thể nhả ra được các đáp án dạng số nguyên/thập phân phức tạp (VD: `13.54`).*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Cloze Simple** | 0.00% | Bế tắc hoàn toàn, chỉ bắn MASK token đơn. |
| **PET-Style** | 0.00% | Vô dụng với câu hỏi tạo trích xuất dài. |
| **Cloze + RAG** | 0.00% | Không có năng lực ghép nối. |
| **Prefix Zero-shot** | 0.00% | Hoàn toàn vô nghĩa với BERT. |
| **Prefix + RAG** | 0.00% | Input RAG vượt dung lượng xử lý tốt nhất. |
| **Prefix + CoT** | 0.00% | Không hỗ trợ chuỗi lý luận. |
| **Decompose + RAG** | 0.00% | Kiến trúc BERT không xử lý tốt Multi-step. |

### 🔍 Demo Prompt (BERT)
```text
Context: [Đoạn văn báo cáo tài chính Apple...]
Question: what was the total volume of sales?
The answer is [MASK].
```

---

## 2. Kết quả trên Kiến trúc Flan-T5 (`google/flan-t5-base`)
*Flan-T5 gánh một chút hy vọng nhờ bản chất Seq2Seq (có khả năng sinh số).*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Prefix + RAG** | **6.67%** | RAG giúp mô hình "copy chép phạt" số liệu khá tốt. |
| **Cloze + RAG** | **6.67%** | Tương tự RAG. |
| **Prefix + CoT** | 3.33% | Suy luận toán học hỏng bét, do T5 quá bé để biết cộng trừ nhân chia. |
| **Cloze Simple** | 0.00% | Bế tắc hoàn toàn (Base model quên format). |
| **PET-Style** | 0.00% | Mô hình không hiểu yêu cầu ngầm. |
| **Prefix Zero-shot** | 0.00% | Hoàn toàn vô định về mặt trích xuất. |
| **Decompose + RAG** | 0.00% | Không hiểu cách tổng hợp số từ logic cắt lẻ. |

### 🔍 Demo Prompt (Flan-T5)
```text
Q: what is the net income in 2012?
A: 554.0

Context: [Báo cáo tài chính]
Q: what is the total equity?
A: <extra_id_0>
```

---

## 3. Kết quả trên Kiến trúc Qwen3 (`Qwen3-0.6B`)
*Kiến trúc Transformer Causal siêu nhỏ không được tinh chỉnh lệnh (Instruct).*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Prefix + RAG** | **3.33%** | Dòng Base chỉ có thể nhại lại Few-shot (RAG) để copy số hú họa. |
| **Cloze + RAG** | **3.33%** | Tương tự Prefix RAG. |
| **Cloze Simple** | 0.00% | Lặp chuỗi / Hallucination nặng. |
| **PET-Style** | 0.00% | Không trích xuất được số thực tế. |
| **Prefix Zero-shot** | 0.00% | Không hiểu format lệnh QA. |
| **Prefix + CoT** | 0.00% | Base model không có khả năng sinh chuỗi giải toán CoT. |
| **Decompose + RAG** | 0.00% | Không hiểu cách ráp số liệu phân mảnh. |

### 🔍 Demo Prompt (Qwen3-0.6B)
```text
Answer the financial question.
Context: [Báo cáo tài chính]
Question: what is the total equity?
Answer: 
```

---

## 4. Kết quả trên Kiến trúc Qwen3.5 (`Qwen3.5-0.8B`)
*Dung lượng lớn hơn một chút (0.8 Tỷ tham số) đem lại thay đổi gì?*

| Phương pháp (Method) | Accuracy | Phân tích |
| :--- | :---: | :--- |
| **Prefix + RAG** | **6.67%** | Đã đuổi kịp Flan-T5, khả năng Copy-Paste Extraction bắt đầu gõ nhịp. |
| **Cloze Simple** | 3.33% | Có hiện tượng vô tình trích xuất trúng vì token độ dài bao phủ tốt hơn. |
| **Cloze + RAG** | 3.33% | Dựa dẫm RAG để trích đoạn context, đôi khi trúng đích. |
| **PET-Style** | 0.00% | Base Qwen chưa học được phrasing style của PET. |
| **Prefix Zero-shot** | 0.00% | Không hiểu cách answer nếu không mồi. |
| **Prefix + CoT** | 0.00% | Sinh "ảo giác" nặng nề khi cộng trừ nhân chia. |
| **Decompose + RAG** | 0.00% | Thất bại ở bước tổng hợp logic sub-question. |

### 🔍 Demo Prompt (Qwen3.5-0.8B)
```text
Analyze this movie review step by step:
Context: [Báo cáo...]
Question: what is the sum?
Let's think step by step to find numbers and compute.
```

---
## 💡 Kết luận Chung (Insight Học Thuật)

1. **Hiệu ứng cái bóng toán học:** Cả 4 mô hình dưới 1B Parameters đều không có tư duy Arithmetic (Toán Số Học). Điều duy nhất giúp chúng ăn điểm (đạt ~6.6%) là kỹ năng **Extraction (Trích xuất) nhờ vào In-context Learning (RAG)** sao chép số lượng giống hệt Prompt đầu vào.
2. **Sự sụp đổ của CoT:** Đi ngược với lý thuyết *Chain-of-Thought* siêu mạnh mẽ trong paper, CoT ở các mô hình siêu nhỏ (Base model) là thảm họa vì sinh ra "Ảo giác Toán Học" (Hallucinations).
3. **MASK Model vô vụng:** Việc Prompt thiết kế cho BERT trên FinQA là lãng phí tài nguyên nghiên cứu vì Task này đòi hỏi đáp án chuỗi (Multi-token generation). 

*(Bản báo cáo này cung cấp cái nhìn thực nghiệm mạnh mẽ nhất chống lại việc "lạm dụng" Prompt dựa trên cảm tính).*
