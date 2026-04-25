# Prompt-Bench: A Multi-Architecture Benchmark for Prompt-based Learning

**Prompt-Bench** đánh giá hệ thống **7 kỹ thuật Prompt Engineering** (Cloze, PET, RAG, CoT, Decompose…) trên **4 kiến trúc mô hình** (BERT, FlanT5, Qwen3-0.6B, Qwen3.5-0.8B) với **4 tập dữ liệu** NLP đa dạng (COPA, RTE, IMDB, FinQA). Mục tiêu: so sánh hiệu quả từng dạng prompt theo kiến trúc mô hình trong bài toán học dựa trên prompt — môn học Statistical Machine Learning (R5).

---

## 🏗️ 1. Cấu trúc Thực nghiệm (Experimental Setup)



### 📚 Tập dữ liệu đọ sức (Datasets)
1. **[RTE] Entailment (Logic Semantic):** Tư duy NLI, nhận dạng hàm ý kéo theo (Premise -> Hypothesis).
2. **[COPA] Commonsense Reasoning:** Kiến thức thường thức xã hội, nguyên nhân & hệ quả thực tế đời sống.

### 🧠 Các Mô hình Ngôn ngữ Khảo sát (Architectures)
* **`bert-base-uncased` (Masked LM):**
* **`google/flan-t5-base` (Seq2Seq):** 
* **`Qwen/Qwen3-0.6B` (Causal LM):** 
* **`Qwen/Qwen3.5-0.8B` (Causal LM):**

### 🎭 Từ điển Kỹ thuật Prompt (Methodologies)
Dự án phân tích 7 loại hình Prompts mang xu hướng ML/NLP:
1. `Cloze Simple`
2. `PET-style` (Pattern-Exploiting Training)
3. `Cloze + RAG` (In-Context Few-shot)
4. `Prefix Zero-shot`
5. `Prefix + RAG`
6. `Prefix + CoT` (Chain-of-Thought Reasoning)
7. `Decompose + RAG` (Multi-Step Logic RAG)

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
├── run_rte_all.py                      <-- Lõi Engine (Universal Runner)
├── run_copa_all.py
├── run_all_models_finqa.py
├── run_qwen_imdb.py
└── README.md
```

---

## 🚀 3. Hướng dẫn chạy Benchmark (Execution Syntax)

Để chạy đánh giá mô hình trên từng tập dữ liệu, bạn có thể thực thi trực tiếp các file script runner ở thư mục gốc. Kết quả sẽ tự động lưu lại logs tại thư mục tương ứng hoặc in ra màn hình.

### Ví dụ chạy từng Dataset Benchmark tổng:
```bash
# Chạy đánh giá tác vụ Entailment (RTE) cho TẤT CẢ model (BERT, T5, Qwen)
python run_rte_all.py

# Chạy đánh giá Commonsense Reasoning (COPA)
python run_copa_all.py

# Chạy đánh giá Finance QA (FinQA)
python run_all_models_finqa.py

# Chạy đánh giá Sentiment Analysis (IMDB)
python run_qwen_imdb.py
```

### ⚠️ Lưu ý về PET-style (Pattern-Exploiting Training)
> [!NOTE]
> Kỹ thuật **PET-style** hiện tại đang trong quá trình được làm lại (Refactoring) để tối ưu và ổn định hơn. Tuy nhiên, nó **đã được tích hợp đầy đủ** vào trong các bộ chạy chính (universal runners). Do đó, khi bạn chạy các lệnh benchmark tổng hợp ở trên, kết quả của PET-style sẽ tự động được kiểm nghiệm và xuất ra cùng với các luận điểm (prompts) khác!
