# 🔄 SML Flow: Sơ đồ Luồng Kiến trúc Thực Nghiệm Prompt Benchmark

Tài liệu này cung cấp bản vẽ trực quan toàn bộ vòng đời (Lifecycle) của 1 hệ thống Benchmark nội bộ được sử dụng trong dự án `SML_master_R5`. Lệnh thực thi sẽ đi từ việc xào nấu Dữ liệu (KB & Eval), qua luồng Bộ định tuyến Prompt (Prompt Router) chia làm 7 ngã, đi vào các hàm dự đoán phân ngành Model, và cuối cùng đối soát sinh ra điểm (Accuracy) ghi vào `.csv`.

## 🎨 Sơ đồ Hoạt động Toàn cảnh (Flowchart)
*Sơ đồ dưới đây được vẽ tự động bằng chuẩn văn bản Mermaid. Bạn có thể sao chép trực tiếp thẻ code này bỏ vào các trình xem Mermaid, hoặc mở xem bằng các Extension hỗ trợ Markdown Preview trên VSCode.*

```mermaid
flowchart TD
    
    %% --- GIAI ĐOẠN 1 ---
    subgraph DataPrep [Giai đoạn 1: Chuẩn bị Dữ liệu & Hệ thống Retrieval]
        A[(Kho Datasets: \nIMDB, FinQA, RTE, COPA)]
        
        A -->|Trích xuất tập Train| B[Mã hóa Vector\nSentenceTransformer\n'all-MiniLM-L6-v2']
        B --> C[(FAISS\nVector Knowledge Base)]
        
        A -->|Cắt mẫu Test/Validation| D([Eval_Data_Subset\nVí dụ: n=100])
    end

    %% --- GIAI ĐOẠN 2 ---
    subgraph PromptModule [Giai đoạn 2: Trạm Tinh chế & Định tuyến Prompt]
        D --> V[Vòng lặp từng Method]
        V --> E{Router:\nKỹ thuật Prompt?}
        
        %% Nhánh Prompt chay
        E -->|Không cần tìm kiếm| F[Tiền xử lý Zero-shot]
        F --> F1(1. Cloze Simple)
        F --> F2(2. PET-Style)
        F --> F3(3. Prefix Zero-shot)
        F --> F4(4. Prefix + CoT)
        
        %% Nhánh Prompt có RAG
        E -->|Kiến thức In-Context| G[Truy vấn Text tương tự]
        C -.-|Cấp ví dụ Few-shot| G
        G --> R1(5. Cloze + RAG)
        G --> R2(6. Prefix + RAG)
        
        %% Nhánh chẻ vấn đề logic Decompose
        E -->|Pipeline 2 Lớp| H[Tư duy Phân rã Vấn đề]
        H --> H1(Gọi Model giải bài toán Phụ\nVD: Trích xuất Context)
        H1 --> H2(Ghép lời giải Phụ + RAG\nvào chung Câu hỏi chính)
        C -.-|Cấp ví dụ| H2
        H2 --> H3(7. Decompose + RAG)
    end
    
    %% --- GIAI ĐOẠN 3 ---
    subgraph Execution [Giai đoạn 3: Tầng Xử lý LLMs Inference]
        F1 & F2 & F3 & F4 & R1 & R2 & H3 --> I([Unified Prompt String\nBiến text hợp nhất])
        
        I --> J{Model Architecture Layer}
        
        J -->|BERT| M1[Masked LM:\nLọc Logits xuất hiện\ncủa target words tại MASK_TOKEN]
        J -->|Flan-T5| M2[Encoder-Decoder:\nGenerate bắt đầu bằng extra_id_0]
        J -->|Qwen| M3[Causal LM:\nAutoregressive Next-Token Generation\ncho đến EOS Token]
        
        M1 & M2 & M3 --> K(String Output Nháp)
    end

    %% --- GIAI ĐOẠN 4 ---
    subgraph Eval [Giai đoạn 4: Đánh giá & Rút trích]
        K --> L[Hàm Đối Sát:\nmatch_answer Regex Regex/Contains]
        L --> M{Khớp với Ground-Truth?}
        M -->|Đúng kiểu Float/Bool/Multiple| O1[Label: TRUE]
        M -->|Trật lất / Sinh ảo giác| O2[Label: FALSE]
        
        O1 & O2 --> P[Save \nresults_folder.csv]
    end
    
    %% Styling and Highlighting Nodes
    style C fill:#9cc2ff,stroke:#333,stroke-width:2px,color:#000
    style P fill:#85e085,stroke:#333,stroke-width:2px,color:#000
    style E fill:#ffcc99,stroke:#333,stroke-width:2px,color:#000
    style J fill:#f2b6b6,stroke:#333,stroke-width:2px,color:#000
```

---

## 📝 Chú giải Hệ thống Tích hợp
1. **In-Context Routing:** Sự kiện mồi dữ liệu `RAG` được xem như một tác vụ nhánh rẽ, không phải mọi Prompt đều gọi RAG. Điều này bảo vệ tốc độ Inference của máy.
2. **Multi-turn Decomposition:** Ở phương pháp thiết kế thứ 7, hệ thống buộc phải "đánh vòng" mũi tên gọi lại Mô hình AI để sinh câu trả lời mồi, sau đó mới lắp vào Prompt chính thức để hỏi lại vòng 2.
3. **Architecture Adapter:** Ở tầng sinh trả lời cuối, AI không được gọi chung bằng hàm `model.generate()`. Nó chia 3 phễu khác nhau với cơ chế hoàn toàn trái ngược (Đo Logits Probability cho BERT vs Auto Generation cho dòng còn lại).
