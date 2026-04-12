import pandas as pd
import torch
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

def load_data(data_path, n_eval=30):
    with open(data_path, "r", encoding="utf-8") as f: data = json.load(f)
    docs = []
    for item in data:
        ctx = " ".join(item.get("pre_text", [])) + " " + " ".join([" ".join(row) for row in item.get("table", [])])
        docs.append({"context": ctx, "question": item["qa"]["question"], "answer": str(item["qa"]["answer"])})
    return docs[n_eval:n_eval+300], docs[:n_eval]

class FAISSRetriever:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model_name)
    def build(self, kb_data):
        self.kb = kb_data
        embeddings = self.model.encode([doc["question"] for doc in kb_data], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return self
    def retrieve(self, query, top_k=2):
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        return [self.kb[i] for i in ids[0]]

def match_answer(pred, true_ans):
    pred = str(pred).lower(); true_ans = str(true_ans).lower()
    if true_ans in pred: return True
    pred_nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
    return true_ans in pred_nums

class FlanT5Prompter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device).eval()

    def process_prompt(self, prompt, max_new_tokens=15):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad(): out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def run_all(self, eval_data, retriever):
        all_results = []
        for method in ["cloze_simple", "pet_style", "cloze_rag", "prefix_zeroshot", "prefix_rag", "prefix_cot", "decompose_rag"]:
            for item in tqdm(eval_data, desc=method):
                ctx = item["context"][:400]; q = item["question"]; target = "<extra_id_0>"
                if method == "cloze_simple": p = f"Context: {ctx}\nQuestion: {q}\nThe answer is {target}"
                elif method == "pet_style": p = f"{ctx}\nBased on table, the answer to '{q}' is {target}"
                elif method == "cloze_rag":
                    s = retriever.retrieve(q, top_k=1)
                    ps = f"Q: {s[0]['question']}\nThe answer is {s[0]['answer']}.\n" if s else ""
                    p = f"{ps}Context: {ctx}\nQ: {q}\nThe answer is {target}"
                elif method == "prefix_zeroshot": p = f"Answer the financial question.\nContext: {ctx}\nQuestion: {q}\nAnswer:"
                elif method == "prefix_rag":
                    s = retriever.retrieve(q, top_k=2)
                    ps = "".join([f"Q: {x['question']}\nA: {x['answer']}\n" for x in s])
                    p = f"{ps}Context: {ctx}\nQ: {q}\nA:"
                elif method == "prefix_cot": p = f"Context: {ctx}\nQ: {q}\nLet's think step by step to find numbers and compute."
                elif method == "decompose_rag":
                    sub = self.process_prompt(f"Sub-question: What numbers are needed for '{q}'?", 15)
                    p = f"Numbers: {sub}. Q: {q}\nFinal Answer:"
                
                if method != "decompose_rag": pred = self.process_prompt(p, 40 if method=="prefix_cot" else 15)
                all_results.append({"method": method, "pred": pred, "true": item["answer"]})
        return all_results

kb_data, eval_data = load_data("/home/tienpv16/Documents/Research/project_SML/SML_master_R5/VuManhCuong/dev.json", 30)
retriever = FAISSRetriever().build(kb_data)
prompter = FlanT5Prompter()
results = prompter.run_all(eval_data, retriever)

out_df = pd.DataFrame(results)
out_df["correct"] = out_df.apply(lambda r: match_answer(r["pred"], r["true"]), axis=1)
print("\n--- FlanT5 Benchmark ---")
for method in out_df["method"].unique():
    acc = out_df[out_df["method"] == method]["correct"].mean()
    print(f"[{method}] Acc (EM-relaxed): {acc:.4f}")
out_df.to_csv("/home/tienpv16/Documents/Research/project_SML/SML_master_R5/FinQA_flant5_benchmark/results_finqa.csv", index=False)
