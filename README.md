# 🚀 SML Master R5: Prompt-based Learning Benchmark Framework

Dự án này là phiên bản đột phá (R5) tiến hành thực nghiệm và đánh giá toàn diện (Benchmark) sức mạnh của **7 Phương pháp Prompt Engineering** tiêu biểu trên **4 Dòng Mô hình Ngôn ngữ đa dạng (LLMs)** đối chiếu qua **4 Tập dữ liệu phức tạp**.

---

## 🏗️ 1. Cấu trúc Thực nghiệm (Experimental Setup)

Hệ thống được thiết kế để đong đếm giới hạn trí tuệ của AI qua việc đa dạng hóa từ Dataset đến Modeling.

### 📚 Tập dữ liệu đọ sức (Datasets)
1. **[IMDB] Sentiment Analysis:** Đọc hiểu cảm xúc văn bản gốc.
2. **[FinQA] Mathematical Reasoning:** Cào dữ liệu tài chính lưới, tính toán và trích xuất số liệu.
3. **[RTE] Entailment (Logic Semantic):** Tư duy NLI, nhận dạng hàm ý kéo theo (Premise -> Hypothesis).
4. **[COPA] Commonsense Reasoning:** Kiến thức thường thức xã hội, nguyên nhân & hệ quả thực tế đời sống.

➡️ *Xem chi tiết tại: [R5 Dataset Configurations](docs/R5_Dataset_Configurations.md)*

### 🧠 Các Mô hình Ngôn ngữ Khảo sát (Architectures)
* **`bert-base-uncased` (Masked LM):** Sức mạnh điền khuyết với cơ chế 2 chiều thuần túy.
* **`google/flan-t5-base` (Seq2Seq):** Kiến trúc Encoder-Decoder được Instruction-Tuned tối đa.
* **`Qwen/Qwen3-0.6B` (Causal LM):** Sức mạnh dự đoán Next-Token với lượng pre-training khổng lồ.
* **`Qwen/Qwen3.5-0.8B` (Causal LM):** Kiến trúc tiên tiến gánh trọn vẹn khả năng tư duy phi tuyến tính ở hệ tham số dưới 1 tỷ.

### 🎭 Từ điển Kỹ thuật Prompt (Methodologies)
Dự án cắm sâu vào phân tích 7 loại hình Prompts mang xu hướng đỉnh cao giới ML/NLP:
1. `Cloze Simple`
2. `PET-Style` (Pattern-Exploiting Training)
3. `Cloze + RAG` (In-Context Few-shot)
4. `Prefix Zero-shot`
5. `Prefix + RAG`
6. `Prefix + CoT` (Chain-of-Thought Reasoning)
7. `Decompose + RAG` (Tư duy chẻ vấn đề Multi-Step)

➡️ *Xem chi tiết tại: [R5 Prompt Methodology Catalog](docs/R5_Prompt_Methodology_Catalog.md)*

---

## 📂 2. Cấu trúc Nguồn Code (Project Structure)

Chúng tôi đã đóng gói 16 Folder độc lập phục vụ cho **Universal Execution**, mỗi folder đều có kịch bản chạy tự động.

```text
SML_master_R5/
├── COPA_bert_benchmark/
├── COPA_flant5_benchmark/
├── COPA_qwen3_0.6b_benchmark/
├── COPA_qwen3_5_0_8b_benchmark/
│
├── FinQA_bert_benchmark/
├── [... Tương tự cho tập FinQA ...]
│
├── IMDB_bert_benchmark/
├── [... Tương tự cho tập IMDB ...]
│
├── RTE_bert_benchmark/
├── [... Tương tự cho tập RTE ...]
│
├── docs/                               <-- Thư mục chứa Báo cáo Khoa Học
│   ├── R5_Architecture_Flowchart.md    <-- Sơ đồ Benchmark Diagram (Mermaid)
│   ├── R5_COPA_Benchmark_Report.md
│   ├── R5_FinQA_Benchmark_Report.md
│   ├── R5_IMDB_Benchmark_Report.md
│   ├── R5_RTE_Benchmark_Report.md
│   ├── R5_Execution_Guide.md           <-- HDSD Môi trường dòng lệnh
│   ├── R5_Dataset_Configurations.md
│   └── R5_Prompt_Methodology_Catalog.md
│
├── run_rte_all.py                      <-- Lõi Engine (Universal Runner)
├── run_copa_all.py
├── run_all_models_finqa.py
├── run_qwen_imdb.py
└── README.md
```

---

## 📖 3. Báo cáo & Khám phá Hàn Lâm (Key Findings)
Tất cả biểu đồ CSV đều được tự động lưu sau cú click chạy. Những phát hiện nổi bật nhất đã được đúc kết sắc bén trong thư mục `docs/`:
- 💡 **FinQA Cảnh Tỉnh:** Khẳng định sự vô vọng của các mô hình cận 1B khi đối mặt với Toán học. Kỹ thuật *Chain-of-Thought (CoT)* thay vì phát huy thế mạnh đã dẫn đến "Ảo giác số học" (Hallucinations) kéo tụt điểm về `0%`.
- 💡 **COPA Đảo Chiều Phép Màu:** Trên nền kiến thức thường thức (Common Sense), bộ đôi `Qwen3.5-0.8B` + `Prefix CoT` biến thành thiên tài khi chạm ngưỡng chính xác **97.00%**! Trong khi cơ chế điền khuyết của BERT sụp đổ quanh mốc 50%.
- 💡 **RTE và Đỉnh cao Decompose:** Việc chẻ nhỏ đối tượng thông tin logic (`Decomposition`) đã lấp lóe rực rỡ với cấu trúc Next-Token, chạm mức **100% Accuracy**.

---

## 🛠 4. Hướng dẫn Chạy (Quick Setup)
Clone Repo, Active Virtual Env, và Run! Toàn bộ hướng dẫn tích hợp thư viện và tự khởi chạy Benchmark đã được cô đọng tại 📄 **[Hướng dẫn Thực Thi Nhanh (Execution Guide)](docs/R5_Execution_Guide.md)**.
