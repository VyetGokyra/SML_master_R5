# Từ điển và Tổng hợp các Kỹ thuật Prompt-based Learning
*Tài liệu này định nghĩa rõ 7 phương pháp Prompt Engineering gốc rễ đã được ứng dụng trong quá trình thực nghiệm đối chiếu trên hệ thống đồ án SML. Những kỹ thuật này được tổng hợp từ paper của Liu et al. (Pre-train, Prompt, Predict) kết hợp cùng trào lưu In-Context RAG hiện đại.*

---

## I. Nhóm Kỹ Thuật Điền Khuyết Đặc Biệt (Cloze-Style Prompts)
*Phương pháp này giả lập lại quá trình "Masked Language Modeling", thường bộc lộ toàn bộ sức mạnh nguyên thủy khi cắm vào kiến trúc Encoder (như BERT) hoặc Causal Auto-regressive (như Qwen).*

### 1. Cloze Simple (Thiết lập khoảng trống nguyên sơ)
**Khái niệm:** Lợi dụng khả năng ngữ pháp tự nhiên của mô hình. Không ra lệnh, chỉ đưa ra văn bản và dừng bút lại ở giữa một chữ nối (như `Because it was...`) để mô hình tự bật ra cảm xúc.
* **Mẫu áp dụng (Masked LM):**
  > Review: "This is a masterpiece of cinema."
  > Because it was **[MASK]**.
* **Mẫu áp dụng (Causal LM):**
  > Premise: The guy turned on the faucet. Choice 1: The water flowed. Choice 2: Total darkness.
  > The correct choice is Choice **(Model tự sinh số 1)**

### 2. PET-Style (Pattern-Exploiting Training)
**Khái niệm:** Thay vì tự do thả nổi, người ta "mớm" sẵn toàn bộ cấu trúc ngôn ngữ của câu trả lời, thiết lập nên "Khuôn mẫu" (Pattern) ngầm giúp model bị khóa mục tiêu. Format này được Schick et al. chứng minh hiệu quả cực cao.
* **Mẫu áp dụng:**
  > [Đoạn nội dung review dài ngoằng...]
  > All in all, the movie was **[MASK]**.
  > *(Sự khác biệt: Nó tổng kết toàn bài thay vì nối nguyên nhân như Cloze Simple).*

### 3. Cloze + RAG (Điền khuyết mớm Few-shot)
**Khái niệm:** Ghép nối phương thức Khôi phục Thông tin (Retrieval Augmented Generation). Hệ thống đi tìm các ví dụ mẫu tương tự ở trong FAISS, ghép chúng thành "Tiền bản", sau đó mới thả đoạn văn bản bị khuyết chờ điền.
* **Mẫu áp dụng:**
  > Review: "Amazing shots..."
  > Sentiment: good.
  > 
  > Review: "The cast is terrible." *(Câu thực tế)*
  > Sentiment: **[MASK]**.

---

## II. Nhóm Kỹ Thuật Chỉ Lệnh Khởi Tạo (Prefix-Style Prompts)
*Phương pháp này ép một mô hình phải hoạt động như Cỗ máy Đối đáp. Rất phù hợp nếu Model đã được Instruction-Tuning (Như Flan-T5) nhưng sẽ là thảm họa cho BERT.*

### 4. Prefix Zero-shot (Hỏi đáp Trực diện)
**Khái niệm:** Diễn đạt dưới dạng một Câu hỏi và ép mô hình phải nhả thẳng câu trả lời theo đúng Format lập trình phía sau bằng lệnh `Answer: `.
* **Mẫu áp dụng (Flan-T5 / Qwen):**
  > Answer the financial question.
  > Context: Apple's net sales were 13.5 million.
  > Question: what was the total volume of sales?
  > Answer: **(Model sinh thẳng số 13.5 million)**

### 5. Prefix + RAG (Hỏi đáp kèm Dữ kiện mồi)
**Khái niệm:** Mang sức mạnh của "In-Context Learning". Ném các ví dụ hỏi đáp đúng/sai trực tiếp vào ngữ cảnh để model bắt chước văn phong (bắt chước chép lại số liệu - Extraction).
* **Mẫu áp dụng:**
  > Premise: The window broke.
  > 1) I threw a rock 2) It rained
  > Q: cause
  > Correct: 1
  > *(Ngăn cách Few-shot)*
  > Premise: [Vấn đề thực tế]
  > 1) [Lựa chọn 1] 2) [Lựa chọn 2]
  > Q: [Loại câu hỏi]
  > Correct: **(Model tự sinh số)*

### 6. Prefix + CoT (Kích hoạt Tư duy Chuỗi khối - Chain of Thought)
**Khái niệm:** Chìa khóa vàng giải quyết mọi vấn đề Common-sense (Thường thức). Ép Model thay vì đọc đáp án ngay (Dễ bị nông cạn) thì phải trải qua nhiều dòng sinh chữ "Think step by step" trước khi chốt lại.
* **Mẫu áp dụng:**
  > Analyze this movie review step by step:
  > Review: "The plot was a total mess but acting saved it."
  > Let's think step by step:
  > **(Model tự phân tích dông dài rồi mới chốt lại Sentiment ở dòng cuối)**

---

## III. Nhóm Kỹ Thuật Phân Rã Hệ Thống (Decomposition Strategies)
*Tách rời các khía cạnh logic để giảm tải Cognitive Load cho Model.*

### 7. Decompose + RAG (Chẻ vấn đề đa mạch)
**Khái niệm:** Phá vỡ logic xử lý 1 nhịp. Thực hiện 2 vòng gọi API Model.
- **Round 1 (Bóc tách Subject):** Gọi model trích riêng thông tin cần tập trung ra khỏi mớ hỗn độn.
- **Round 2 (Đánh giá Subject):** Đưa Subject ròng đó vào chung với câu hỏi gốc và RAG của FAISS để hỏi đáp án cuối cùng.
* **Mẫu áp dụng:**
  - **Lượt chạy ngầm của máy 1:**
    > Hypothesis: Weapons of Mass Destruction Found in Iraq.
    > Question: What is the main action? -> **(Máy sinh lén đáp án: Find weapons)**
  - **Lượt chạy chốt sổ tự động 2:**
    > Main action: Find weapons
    > Premise: No Weapons of Mass Destruction Found in Iraq Yet.
    > Hypothesis: Weapons of Mass Destruction Found in Iraq.
    > Does the premise entail the hypothesis? **Wait for Generation or [MASK]**
