# Prompt-Bench: A Multi-Architecture Benchmark for Prompt-based Learning

**Prompt-Bench** đánh giá hệ thống **6 kỹ thuật Prompt Engineering** (Cloze, PET, RAG, CoT, Decompose…) trên **4 kiến trúc mô hình** (BERT, FlanT5, Qwen3-0.6B, Qwen3.5-0.8B) với **4 tập dữ liệu** NLP đa dạng (COPA, RTE, IMDB, FinQA). Mục tiêu: so sánh hiệu quả từng dạng prompt theo kiến trúc mô hình trong bài toán học dựa trên prompt — môn học Statistical Machine Learning (R5).

---

## 🏗️ 1. Cấu trúc Thực nghiệm (Experimental Setup)

Hệ thống được thiết kế để đong đếm giới hạn trí tuệ của AI qua việc đa dạng hóa từ Dataset đến Modeling.

### 📚 Tập dữ liệu đọ sức (Datasets)
1. **[RTE] Entailment (Logic Semantic):** Tư duy NLI, nhận dạng hàm ý kéo theo (Premise -> Hypothesis).
2. **[COPA] Commonsense Reasoning:** Kiến thức thường thức xã hội, nguyên nhân & hệ quả thực tế đời sống.

### 🧠 Các Mô hình Ngôn ngữ Khảo sát (Architectures)
* **`bert-base-uncased` (Masked LM):** Sức mạnh điền khuyết với cơ chế 2 chiều thuần túy.
* **`google/flan-t5-base` (Seq2Seq):** Kiến trúc Encoder-Decoder được Instruction-Tuned tối đa.
* **`Qwen/Qwen3-0.6B` (Causal LM):** Sức mạnh dự đoán Next-Token với lượng pre-training khổng lồ.
* **`Qwen/Qwen3.5-0.8B` (Causal LM):** Kiến trúc tiên tiến gánh trọn vẹn khả năng tư duy phi tuyến tính ở hệ tham số dưới 1 tỷ.

### 🎭 Từ điển Kỹ thuật Prompt (Methodologies)
Dự án cắm sâu vào phân tích 6 loại hình Prompts mang xu hướng đỉnh cao giới ML/NLP:
1. `Cloze Simple`
2. `Cloze + RAG` (In-Context Few-shot)
3. `Prefix Zero-shot`
4. `Prefix + RAG`
5. `Prefix + CoT` (Chain-of-Thought Reasoning)
6. `Decompose + RAG` (Multi-Step Logic RAG)

---

## 📂 2. Cấu trúc Nguồn Code (Project Structure)


```text
SML_master_R5/
├── COPA_bert_benchmark/
├── COPA_flant5_benchmark/
├── COPA_qwen3_0.6b_benchmark/
├── COPA_qwen3_5_0_8b_benchmark/
│
├── RTE_bert_benchmark/
├── [... Tương tự cho tập RTE ...]
│
├── docs/                               <-- Thư mục chứa Báo cáo Khoa Học
│   ├── R5_Architecture_Flowchart.md    <-- Sơ đồ Benchmark Diagram (Mermaid)
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

