# Báo cáo Đánh giá Benchmark các Kỹ thuật Prompt trên Dataset IMDB

Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm **7 phương pháp Prompt-based Learning** (liên kết với paper Liu et al. & Min et al.) chạy trên hai dòng kiến trúc ngôn ngữ lớn: **BERT (Masked LM)** và **Flan-T5 (Generative LM)** đối với tác vụ phân loại cảm xúc (Sentiment Analysis) sử dụng dataset IMDB.

---

## 1. Kết quả trên Kiến trúc BERT (`bert-base-uncased`)

BERT là một mô hình Masked LM, có hiệu suất tốt nhất khi thực hiện các tác vụ **Fill-in-the-blank (Cloze)** thay vì sinh (Generate) văn bản. Do đó, các kỹ thuật "Prefix" buộc phải được chạy thông qua việc mô phỏng một khoảng trống `[MASK]` ở cuối text.

### 📊 Bảng Kết quả Benchmark (BERT)

| Phương pháp (Method) | Accuracy | F1-Score | Phân tích |
| :--- | :---: | :---: | :--- |
| **Cloze Simple** | **66.00%** | **0.7119** | **Tốt nhất trên BERT:** Giữ đúng lối pre-training gốc. |
| **Prefix + RAG** | 53.00% | 0.6569 | Cải thiện nhờ Few-shot context mồi. |
| **PET-Style** | 51.00% | 0.6423 | Tương đối giống Cloze Simple nhưng phrasing làm khó BERT. |
| **Cloze + RAG** | 49.00% | 0.6434 | RAG dài quá (vượt 512 tokens input) làm phân tán MASK. |
| **Decompose + RAG** | 47.00% | 0.6345 | BERT không tính toán Logic tổng hợp nhiều Aspect tốt. |
| **Prefix Zero-shot**| 46.00% | 0.6301 | BERT không có khả năng hiểu Instruction-tuning. |
| **Prefix + CoT** | 46.00% | 0.6301 | Tương tự Zero-shot, BERT không thể đọc chuỗi suy luận. |

### 🔍 Demo Prompts (BERT)

> Quy ước: Output không gian (Label Space) là `['good', 'bad']`. Model sẽ dự đoán xếp hạng xác suất `P(good | MASK)` và `P(bad | MASK)`. 

**1. Cloze Simple**
```text
[Bản review của đạo diễn Christopher Nolan về phim...]
Because it was [MASK].
```

**2. PET-Style**
```text
[Bản review...]
All in all, the movie was [MASK].
```

**3. Cloze + RAG**
```text
Review: This movie is a masterpiece...
Sentiment: good.

Review: [Bản review thực tế cần phân tích...]
Sentiment: [MASK].
```

**4. Prefix Zero-shot (Adapted cho BERT)**
```text
Review: [Bản review...]
Sentiment: [MASK].
```

**5. Prefix + RAG (Adapted cho BERT)**
```text
Context: 
Review: Great acting...
Sentiment: good.

Review: [Bản review thực tế...]
Sentiment: [MASK].
```

**6. Prefix + CoT (Adapted cho BERT)**
```text
Review: [Bản review...]
Let's think step by step. The review contains positive and negative points. Overall, it is [MASK].
```

**7. Decompose + RAG (Adapted cho BERT)**
*(Chạy 3 lượt dự đoán riêng biệt, sau đó dùng thuật toán lấy đa số ở bên ngoài)*
```text
Lượt 1: Review: [...] Question: How is the acting? The acting is [MASK].
Lượt 2: Review: [...] Question: How is the plot? The plot is [MASK].
Lượt 3: Review: [...] Question: How is the directing? The directing is [MASK].
=> Tổng hợp lại nếu Good >= 2 vote thì Overall Positive.
```

---

## 2. Kết quả trên Kiến trúc Flan-T5 (`google/flan-t5-base`)

Flan-T5 là một mô hình Generative Encoder-Decoder đã được tinh chỉnh qua Instruction (Instruction-Tuning). Mô hình cực kỳ nhạy với các câu lệnh trực tiếp sinh ra output (`Prefix`). Khác với BERT, biểu diễn một prompt Cloze lên Flan-t5 bắt buộc phải sử dụng mã thông báo `<extra_id_0>`.

### 📊 Bảng Kết quả Benchmark (Flan-T5)

| Phương pháp (Method) | Accuracy | F1-Score | Phân tích |
| :--- | :---: | :---: | :--- |
| **Prefix Zero-shot**| **90.00%** | **0.9046** | **Tốt nhất trên Flan-T5:** Nó hoàn toàn phục tùng Instruction-tuning. |
| **Prefix + RAG** | **90.00%** | **0.9000** | RAG trên tác vụ Classification cơ bản không tăng thêm lợi ích nào. |
| **Decompose + RAG** | 78.00% | 0.7789 | Bắt nó cắt nhỏ khía cạnh rồi đánh giá tạo rủi ro lặp ngữ nghĩa cao. |
| **Cloze + RAG** | 63.00% | 0.7462 | Rất tốt trong hệ Cloze nhờ học được Context mồi RAG đi kèm. |
| **Prefix + CoT** | 34.00% | 0.4228 | Hallucinations phá vỡ hoàn toàn chuỗi văn bản (T5-Base quá yếu để CoT). |
| **PET-Style** | 2.00% | 0.0383 | Model sinh ra nội dung lung tung không khớp không gian Output. |
| **Cloze Simple** | 0.00% | 0.0000 | Tương tự PET-Style, model mất phương hướng với `<extra_id_0>`. |

### 🔍 Demo Prompts (Flan-T5)

> Quy ước: Model sẽ tự do sinh Text (Generation). Label hy vọng trích xuất được là `['positive', 'negative']`.

**1. Prefix Zero-shot**
```text
Review: "[Bản review...]"
Is this movie review positive or negative?
Answer:
```

**2. Prefix + RAG**
```text
Review: "Brilliant screenplay and acting..."
Sentiment: Positive

Review: "[Bản review thực tế...]"
Is this movie review positive or negative?
Sentiment:
```

**3. Prefix + CoT**
```text
Analyze this movie review step by step:
Review: "[Bản review...]"
Step 1: Identify key positive words.
Step 2: Identify key negative words.
Step 3: Conclude the overall sentiment (Positive or Negative).
Analysis:
```

**4. Decompose + RAG**
*(Chia nhỏ quá trình thành 2 nhịp sinh văn bản trên Flan-T5)*
```text
Nhịp 1 (Sinh các khía cạnh cần phân tích đánh giá):
List 3 main aspects discussed in this movie review (e.g., acting, plot, directing). Separate them by commas.
Review: "[Bản review...]"
Aspects: 
-> (Giả sử model tạo ra: acting, soundtrack)

Nhịp 2 (Loop qua từng Aspect để đánh giá rồi tổng hợp votes):
Review: "The acting was terrible but the soundtrack was decent" -> Sentiment: Negative
Review: "[Bản review thực tế]"
What is the sentiment about 'acting' in the review? (Positive/Negative/Neutral)
Answer:
```

**5. Cloze Simple (Adapted cho Flan-T5)**
```text
[Bản review...]
Because it was <extra_id_0>.
```

**6. PET-Style (Adapted cho Flan-T5)**
```text
[Bản review...]
All in all, the movie was <extra_id_0>.
```

**7. Cloze + RAG (Adapted cho Flan-T5)**
```text
Review: "Amazing shots..."
Sentiment: good.

Review: "[Bản review thực tế...]"
Sentiment: <extra_id_0>.
```

---

## 3. Kết quả trên Kiến trúc Qwen (Causal LM - Decoder Only)

Nhóm kiến trúc Qwen (`Qwen3-0.6B` và `Qwen3.5-0.8B`) là các mô hình Causal LM (sinh văn bản một chiều) có kích thước siêu nhỏ (dưới 1 tỷ tham số). Phân khúc này thường chỉ có khả năng tiếp nối logic tự động chứ không phải điền khuyết (Cloze) hay bị ngợp trước Instruction quá phức tạp.

### 📊 Bảng Kết quả Benchmark (Qwen3-0.6B)

| Phương pháp (Method) | Accuracy | F1-Score | Phân tích |
| :--- | :---: | :---: | :--- |
| **Prefix Zero-shot** | **91.00%** | **0.9101** | Bất ngờ lớn! Dòng này tiếp nối cực chuẩn xác văn phong Review. |
| **Prefix + RAG** | 67.00% | 0.6606 | RAG khiến Context quá dài, làm model nhỏ vỡ nát Attention. |
| **Prefix + CoT** | 53.00% | 0.5320 | Model quá yếu để sinh chuỗi lý luận (bị lặp hoặc sinh sai ngữ pháp). |
| **Decompose + RAG** | 53.00% | 0.5050 | Điểm thấp tụt vì bóc tách aspect vượt quá năng lực lý luận. |
| **Cloze + RAG** | 3.00% | 0.0568 | Thất bại do không thể hiểu khoảng trống sau chữ `Because it was`. |
| **PET-Style** | 0.00% | 0.0000 | Model sinh tràn lan văn bản lung tung thay vì nhả word. |
| **Cloze Simple** | 0.00% | 0.0000 | Tương tự PET-Style, model sinh các cụm câu vô nghĩa. |

### 📊 Bảng Kết quả Benchmark (Qwen3.5-0.8B)

| Phương pháp (Method) | Accuracy | F1-Score | Phân tích |
| :--- | :---: | :---: | :--- |
| **Decompose + RAG** | **66.00%** | **0.6605** | Bản 0.8B xử lý chunk information tốt hơn hẳn bản 0.6B. |
| **Prefix + CoT** | 52.00% | 0.4170 | Giữ mức sàn, bị vướng hallucination. |
| **Prefix + RAG** | 50.00% | 0.3811 | RAG không hiệu quả với context window nhỏ bị nhồi nhét. |
| **Cloze + RAG** | 48.00% | 0.5924 | Cải thiện ngoạn mục so với 0.6B khi điền tiếp ngữ cảnh. |
| **PET-Style** | 19.00% | 0.2928 | Đã bắt đầu phát sinh đúng format từ ngữ. |
| **Prefix Zero-shot** | 0.00% | 0.0000 | Thất bại do bản Base lặp token không nín được. |
| **Cloze Simple** | 0.00% | 0.0000 | Trôi token. |

### 🔍 Demo Prompts (Qwen Causal LM)

> Quy ước: Causal LM tự do dự đoán Next-Token. Hàm sẽ chặn token khi đạt độ dài quy định và kiểm tra chuỗi xem có `good/bad/positive/negative` không.

**1. Mẫu Prefix chung (Zero-shot, CoT)**
```text
Analyze this movie review step by step:
Review: "[Bản review...]"
Step 1: Identify key positive words.
...
Analysis: [Qwen tự viết tiếp từ đây]
```

**2. Mẫu Cloze chung (Adapted cho Causal Generation)**
```text
Review: "[Bản review...]"
Because it was [Qwen tự viết tiếp từ đây mong đợi nhả cụm từ good/bad/terrible]
```

---
*Báo cáo được trích xuất dựa trên bộ Source code thống nhất thuộc kho lưu trữ project_SML/SML_master_R5. Kết quả thực nghiệm chuẩn xác đại diện cho 3 kiến trúc: Masked LM (BERT), Seq2Seq (Flan-T5) và Decoder-Only (Qwen).*
