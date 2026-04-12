import os
import csv
import json
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Setup Environment - USE YOUR ACTUAL API KEY
os.environ["GOOGLE_API_KEY"] = "GOOGLE_API_KEY"

# Step 1: Data Ingestion (Matches your folder structure)
try:
    with open("dev.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:50]
except FileNotFoundError:
    print("Error: dev.json not found. Check your directory!")
    exit(1)

documents = []
for item in data:
    qa_block = item.get("qa", {})
    combined_content = f"{item.get('pre_text', '')}\n{json.dumps(item.get('table', {}))}\n{item.get('post_text', '')}"
    metadata = {"question": qa_block.get("question", ""), "answer": str(qa_block.get("answer", ""))}
    documents.append(Document(page_content=combined_content, metadata=metadata))

# Step 2: Initialize LLM with the most stable string
# Removed 'models/' prefix which often causes the 'v1beta' error
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

cot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="You are a financial analyst. Answer with step-by-step math.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
)

# Step 3: Chains
baseline_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
baseline_chain = RetrievalQA.from_chain_type(llm=llm, retriever=baseline_retriever, chain_type="stuff", chain_type_kwargs={"prompt": cot_prompt})

adv_retriever = MultiQueryRetriever.from_llm(retriever=baseline_retriever, llm=llm)
advanced_chain = RetrievalQA.from_chain_type(llm=llm, retriever=adv_retriever, chain_type="stuff", chain_type_kwargs={"prompt": cot_prompt})

# Step 4: Execution
benchmark_indices = [0, 15, 28, 34, 48]
csv_path = "benchmark_results.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["Index", "Question", "Ground Truth", "Baseline Answer", "Advanced Answer"])
    writer.writeheader()

    for idx in benchmark_indices:
        item = data[idx]
        question = item.get("qa", {}).get("question", "No Question")
        ground_truth = str(item.get("qa", {}).get("answer", "No Answer"))
        
        print(f"\n[Testing Index {idx}]...")
        
        # --- Baseline Test ---
        try:
            b_res = baseline_chain.invoke({"query": question})
            b_ans = b_res.get("result")
        except Exception as e:
            b_ans = f"Baseline Error: {str(e)}"
            print(f"!!! Baseline Failed: {e}")

        # --- Advanced Test ---
        try:
            a_res = advanced_chain.invoke({"query": question})
            a_ans = a_res.get("result")
        except Exception as e:
            a_ans = f"Advanced Error: {str(e)}"
            print(f"!!! Advanced Failed: {e}")

        writer.writerow({
            "Index": idx, "Question": question, "Ground Truth": ground_truth,
            "Baseline Answer": b_ans, "Advanced Answer": a_ans
        })
        csv_file.flush()

print(f"\nDone. Results saved to {csv_path}")