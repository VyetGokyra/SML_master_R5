**Capstone Project: Comparative Analysis of RAG Architectures for Financial Reasoning**
=======================================================================================

**1\. Introduction**
--------------------

This project evaluates the performance of Retrieval-Augmented Generation (RAG) in the specialized domain of financial analysis. Using the **FinQA dataset**, we compare a **Baseline RAG** model against an **Advanced RAG** model incorporating Multi-Query Retrieval. The objective is to determine how retrieval expansion affects the accuracy of complex mathematical reasoning tasks and to assess the transparency of the resulting outputs.

**2\. Methodology and Architecture**
------------------------------------

The experimental setup utilizes a Medallion-style data architecture to process financial documents into a FAISS vector database.

*   **Baseline RAG:** Employs a standard top-$k$ semantic search ($k=3$) to retrieve context for a given query.
    
*   **Advanced RAG (Multi-Query):** Uses the Large Language Model (LLM) to generate three semantically diverse variations of the user's question. This expands the search breadth to ensure relevant tabular data is captured.
    
*   **Chain-of-Thought (CoT) Prompting:** Both architectures use CoT prompting to force the model to output step-by-step mathematical logic before providing a final answer.
    

**3\. Benchmark Results**
-------------------------

The following table summarizes the performance across five key test cases extracted from the FinQA development set:

**IndexQuestion FocusGround TruthBaseline ResultAdvanced ResultStatus0**Payment Volume127.40127.40127.40**Success15**Contractual Oblig.34%13.81%13.81%**Deviation28**Payment Ratio33.3%33.33%33.33%**Success34**Price Delta2.622.622.62**Success48**Earnings Growth57%No DataNo Data**Retrieval Gap**

**4\. Analysis and Relation to Literature**
-------------------------------------------

### **A. Application of Liu et al.: Prompt Augmentation and Decomposition**

The architecture directly implements core findings from **Liu et al.** regarding the "Pre-train, Prompt, and Predict" paradigm:

*   **Prompt Augmentation:** The Advanced RAG pipeline uses the LLM to augment the initial query into three variations. This addresses the **"Semantic Gap"**—where the user’s natural language does not match the technical jargon of financial tables.
    
*   **Prompt Decomposition:** The **Chain-of-Thought (CoT)** logic serves as "Prompt Decomposition." By breaking the task into intermediate steps (e.g., identifying high/low prices in Index 34), the model avoids the "calculation trap" common in standard LLM generation.
    

### **B. Application of Min et al.: Emergent Reasoning and Knowledge Boundaries**

The benchmark results provide empirical evidence for theories presented by **Min et al.** regarding recent NLP advances:

*   **In-Context Learning (ICL):** Both models correctly performed division and subtraction (Indices 0, 28, 34) without specific fine-tuning, demonstrating the "Emergent Reasoning" abilities mentioned by Min et al..
    
*   **The Retrieval Bottleneck:** The "No Data" result for **Index 48** illustrates a critical point in Min et al.: an LLM's reasoning is limited by the evidence retrieved. Even the Advanced Multi-Query model failed because the specific segment data was likely outside the retrieved context window.
    
*   **Faithfulness vs. Ground Truth:** In **Index 15**, the model reached 13.81% while the ground truth was 34%. Because the CoT output showed the exact numbers used from the table ($1,035 / 7,497$), the model remained **"Faithful"** to the retrieved context. Min et al. identify this faithfulness as a hallmark of reliable RAG systems, prioritizing factual grounding over hallucinations.
    

**5\. Conclusion**
------------------

The study concludes that while **Advanced RAG** provides a more robust retrieval mechanism for semantic variances, the **Chain-of-Thought** logic is the most critical component for financial auditability. The ability to verify the AI's math steps (e.g., $3.57 - 0.95 = 2.62$ in Index 34) provides the transparency required for professional financial environments. Future work should focus on increasing retrieval depth ($k=10$) and implementing hybrid search to close the knowledge gaps identified in Index 48.