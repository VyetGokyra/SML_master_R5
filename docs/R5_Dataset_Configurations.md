# Tài liệu Đặc tả & Cấu hình Kỹ thuật của Các Dataset
*Tài liệu này mô tả chi tiết nguồn gốc, số liệu mẫu và chính sách tùy biến hệ thống trích chọn dữ liệu phục vụ riêng cho đồ án SML_master_R5 phân tích Prompt-based Learning.*

Trong toàn bộ quá trình thực nghiệm, để tránh tình trạng "Over-fitting" kiến thức của một miền cụ thể hoặc "Thiên vị Kiến trúc", hệ thống đánh giá đã sử dụng tới **4 Tập dữ liệu (Datasets)** chéo mang phổ tri thức khác biệt hoàn toàn: **Cảm tính (IMDB) -> Suy luận Toán Học (FinQA) -> Logic Suy Diễn (RTE) -> Thường Thức Xã Hội (COPA).**

Dưới đây là chi tiết cụ thể:

---

## 1. Tập Dữ liệu IMDB (Phân tích Cảm Xúc)
* **Nguồn/Source:** [HuggingFace Hub](https://huggingface.co/datasets/imdb) (Bắt nguồn từ Đại học Stanford nghiên cứu *Large Movie Review Dataset*).
* **Code Tải (Load):** `load_dataset("imdb")`
* **Mô tả (Description):** Kho tàng đánh giá phim. Câu hỏi phân loại nhị phân chuẩn mực (Binary Classification). Model cần nhận biết được đoạn review phim này là tích cực (Positive - Label 1) hay tiêu cực (Negative - Label 0). 
* **Quy mô Sample gốc:** `25,000` dòng Train, `25,000` dòng Test.
* **Cấu hình Benchmark (Setting):**
   * Do đặc thù độ dài tự do của review khá lớn, cấu hình hệ thống thiết lập giới hạn input cắt đi những phần văn bản sau kỷ lục `1500 ký tự`.
   * **Setup Kiến thức RAG:** Trích lập một kho `Knowledge Base (KB)` ngẫu nhiên gồm `2000` văn bản lấy trực tiếp từ tập Train. Gắn vector vào FAISS.
   * **Tập Đánh giá:** Random `n_eval = 100` mẫu đại diện trích từ tập *Test split* để đo lường. Label `['good', 'bad']` hoặc `['positive', 'negative']` được thiết lập làm từ khóa dự đoán chuẩn cho tất cả LLMs.

---

## 2. Tập Dữ liệu FinQA (Hiểu Toán Tài Chính)
* **Nguồn/Source:** Bắt nguồn từ Paper [FinQA (Chen et al., 2021)](https://github.com/czyssrs/FinQA). Trong máy chạy thực nghiệm được Load từ File Local Dump có sẵn.
* **Code Tải (Load):** `json.load(open('VuManhCuong/dev.json'))`
* **Mô tả (Description):** Hệ thống Question Answering cực nằng nề của giới chuyên môn Tài Chính. Bao gồm `pre_text` và `post_text` (lời bình luận văn bản) cùng `table` (dữ liệu lưới tính toán). LLMs phải trích xuất hoặc tự kết nối số học để ra được đáp án `answer` dạng số chính xác (VD: `52.0`, `127.4`).
* **Quy mô Sample gốc:** Khoảng `~8,000` mẫu toàn cục. Trong cấu hình file `dev.json` lưu giữ hàng ngàn mẫu test.
* **Cấu hình Benchmark (Setting):**
   * Hệ thống tự động thiết kế lại Format cho Model: Gộp văn bản và nối bảng bằng ký tự | để phẳng hóa dữ liệu lưới. Giới hạn `800-1000 characters` để tránh nhiễu OOM (Over-memory).
   * **Setup Kiến thức RAG:** Từ danh sách `dev.json`, cắt `300` mẫu cuối cùng băm vào FAISS làm tệp mồi Few-shot.
   * **Tập Đánh giá:** Set `n_eval = 30` mẫu thử cứng trên đỉnh (đảm bảo ko trùng với RAG). Metric kiểm tra độ chính xác là *EM-relaxed (Exact Match có châm chước)* – Đo xem chuỗi số (digits) được trích xuất/sinh ra có nằm trong câu trả lời nguyên bản của Dataset hay không.

---

## 3. Tập Dữ liệu RTE (Nhận dạng Suy luận Ngữ nghĩa)
* **Nguồn/Source:** [HuggingFace SuperGLUE Benchmark](https://huggingface.co/datasets/super_glue) (Recognizing Textual Entailment).
* **Code Tải (Load):** `load_dataset("super_glue", "rte")`
* **Mô tả (Description):** Cung cấp 2 câu: `Premise` (Bối cảnh) và `Hypothesis` (Giả thuyết luận). Mục tiêu phải nhận diện hàm ý kéo theo: Bối cảnh có bao hàm định lý này không? Label `0`: Entailment (Suy ra được -> "yes"). Label `1`: Not Entailment (Phép suy luận sai -> "no").
* **Quy mô Sample gốc:** `2,490` Train / `277` Validation / `3,000` Test.
* **Cấu hình Benchmark (Setting):**
   * **Setup Kiến thức RAG:** Hút dung lượng khổng lồ. Vét sạch `2,490` bản ghi của tập Train để ngâm vào FAISS.
   * **Tập Đánh giá:** Không dùng tập test (do HF khóa nhãn). Chạy trực tiếp qua List của `Validation Split` với `n_eval = 100` câu đầu tiên chuẩn mực. Output mapping cứng là cặp từ `"yes"/"no"`.

---

## 4. Tập Dữ liệu COPA (Suy luận Thường thức Đời sống)
* **Nguồn/Source:** [HuggingFace SuperGLUE Benchmark](https://huggingface.co/datasets/super_glue) (Choice Of Plausible Alternatives).
* **Code Tải (Load):** `load_dataset("super_glue", "copa")`
* **Mô tả (Description):** Tuyệt tác đo đạc trình độ "Khôn vặt đời sống" (Common Sense) của AI. Cho một sự cố `Premise`. Hỏi `Question` xem bạn chọn `Choice 1` hay `Choice 2` là lý do gây ra sự cố đó (cause) hoặc kết quả của sự cố đó (effect).
* **Quy mô Sample gốc:** Cực kỳ nhỏ gọn: `400` Train / `100` Validation.
* **Cấu hình Benchmark (Setting):**
   * Hệ thống tự động phân nhánh chữ lót Prompt: Nếu question là `cause` -> Nối câu bằng chữ `"because"`. Nếu question là `effect` -> Nối câu bằng chữ `"therefore" / "so"`.
   * **Setup Kiến thức RAG:** Ăn trọn vẹn `400` dòng của tập Train đưa vào Kho mồi Retrieval.
   * **Tập Đánh giá:** Dùng sạch sành sanh `100` mẫu của tập Validation để chạy điểm Evaluation. Model chốt phải trả lời kết quả trúng đích là `"1"` hoặc `"2"`. Trang bị Regex kiểm tra nếu sinh ra chữ `"first / one"` hoặc `"second / two"`.
