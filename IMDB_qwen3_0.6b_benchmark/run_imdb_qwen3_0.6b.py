"""
IMDB Benchmark for Qwen Causal LMs
"""
import argparse
import pandas as pd
import torch
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class IMDBDataLoader:
    def __init__(self, n_kb=2000, n_eval=100, seed=42):
        self.n_kb = n_kb
        self.n_eval = n_eval
        self.seed = seed

    def load(self):
        print(f"Loading IMDB dataset (KB: {self.n_kb}, Eval: {self.n_eval})...")
        ds = load_dataset("imdb")
        
        import random; random.seed(self.seed)
        train_ids = random.sample(range(len(ds["train"])), self.n_kb)
        test_ids = random.sample(range(len(ds["test"])), self.n_eval)
        
        kb = [(ds["train"][i]["text"], "positive" if ds["train"][i]["label"] == 1 else "negative") 
              for i in train_ids]
        eval_ = [(ds["test"][i]["text"], "positive" if ds["test"][i]["label"] == 1 else "negative") 
                 for i in test_ids]
        
        return kb, eval_


class FAISSRetriever:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self.model = None
        self.index = None
        self.docs = []

    def build(self, texts):
        print(f"Building FAISS index for {len(texts)} documents...")
        self.model = SentenceTransformer(self.embed_model_name)
        embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.docs = list(texts)
        return self

    def retrieve(self, query, top_k=2):
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        return [(self.docs[i], float(scores[0][j])) for j, i in enumerate(ids[0])]


class QwenPromptClassifier:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=10):
        # We need to make sure we pad correctly if tokenizer demands it
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        return text

    # ---- Generative Adaptations ----
    
    def run_prefix_zeroshot(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix Zero-shot"):
            short_text = text[:1500]
            prompt = f"Review: \"{short_text}\"\nIs this movie review positive or negative?\nAnswer:"
            output = self.generate(prompt, max_new_tokens=5)
            pred = "positive" if "positive" in output else "negative" if "negative" in output else "unknown"
            results.append({"method": "prefix_zeroshot", "pred": pred, "true": label})
        return results

    def run_prefix_rag(self, eval_data, retriever, kb_dict):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix + RAG"):
            retrieved = retriever.retrieve(text, top_k=3)
            shots = ""
            for doc, _ in retrieved:
                doc_lbl = kb_dict[doc]
                shots += f'Review: "{doc[:150]}..."\nSentiment: {doc_lbl.capitalize()}\n\n'
                
            short_text = text[:1000]
            prompt = f"{shots}Review: \"{short_text}\"\nIs this movie review positive or negative?\nSentiment:"
            output = self.generate(prompt, max_new_tokens=5)
            pred = "positive" if "positive" in output else "negative" if "negative" in output else "unknown"
            results.append({"method": "prefix_rag", "pred": pred, "true": label})
        return results

    def run_prefix_cot(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix + CoT output"):
            short_text = text[:1000]
            prompt = (
                f"Analyze this movie review step by step:\n"
                f"Review: \"{short_text}\"\n"
                f"Step 1: Identify key positive words.\n"
                f"Step 2: Identify key negative words.\n"
                f"Step 3: Conclude the overall sentiment (Positive or Negative).\n"
                f"Analysis:"
            )
            output = self.generate(prompt, max_new_tokens=60)
            
            if "positive" in output and "negative" not in output:
                pred = "positive"
            elif "negative" in output and "positive" not in output:
                pred = "negative"
            else:
                pos_idx = output.rfind("positive")
                neg_idx = output.rfind("negative")
                if pos_idx > neg_idx: pred = "positive"
                elif neg_idx > pos_idx: pred = "negative"
                else: pred = "unknown"
                    
            results.append({"method": "prefix_cot", "pred": pred, "true": label})
        return results

    def run_decompose_rag(self, eval_data, retriever, kb_dict):
        results = []
        for text, label in tqdm(eval_data, desc="Decompose + RAG"):
            short_text = text[:800]
            decomp_prompt = f"List 3 main aspects discussed in this movie review (e.g., acting, plot, directing). Separate them by commas.\nReview: \"{short_text}\"\nAspects:"
            aspects_output = self.generate(decomp_prompt, max_new_tokens=20)
            aspects = [a.strip() for a in aspects_output.replace(".", ",").split(",") if a.strip()]
            if not aspects:
                aspects = ["overall"]
            else:
                aspects = aspects[:3]
                
            aspect_sentiments = []
            for aspect in aspects:
                query = f"{aspect} sentiment: {short_text[:100]}"
                examples = retriever.retrieve(query, top_k=2)
                shots = ""
                for doc, _ in examples:
                    shots += f'Review: "{doc[:100]}..." -> Sentiment: {kb_dict[doc].capitalize()}\n'
                
                aspect_prompt = (
                    f"{shots}\n\nReview: \"{short_text[:400]}\"\n"
                    f"What is the sentiment about '{aspect}' in the review? (Positive/Negative/Neutral)\nAnswer:"
                )
                ans = self.generate(aspect_prompt, max_new_tokens=5)
                aspect_sentiments.append(ans)
                
            pos_votes = sum(1 for s in aspect_sentiments if "positive" in s)
            neg_votes = sum(1 for s in aspect_sentiments if "negative" in s)
            pred = "positive" if pos_votes >= neg_votes else "negative"
            
            results.append({"method": "decompose_rag", "pred": pred, "true": label})
        return results

    # ---- Cloze Adaptations for Causal LM ----
    def run_cloze_simple(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Cloze Simple"):
            short_text = text[:1500] 
            prompt = f"Review: {short_text}\nBecause it was "
            output = self.generate(prompt, max_new_tokens=5)
            pred = "positive" if ("good" in output or "positive" in output or "great" in output) else \
                   "negative" if ("bad" in output or "negative" in output or "terrible" in output) else "unknown"
            results.append({"method": "cloze_simple", "pred": pred, "true": label})
        return results

    def run_pet_style(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="PET-style"):
            short_text = text[:1500]
            prompt = f"Review: {short_text}\nAll in all, the movie was "
            output = self.generate(prompt, max_new_tokens=5)
            pred = "positive" if ("good" in output or "positive" in output or "great" in output) else \
                   "negative" if ("bad" in output or "negative" in output or "terrible" in output) else "unknown"
            results.append({"method": "pet_style", "pred": pred, "true": label})
        return results

    def run_cloze_rag(self, eval_data, retriever, kb_dict):
        results = []
        for text, label in tqdm(eval_data, desc="Cloze RAG"):
            retrieved = retriever.retrieve(text, top_k=2)
            context = ""
            for doc, _ in retrieved:
                doc_map = "good" if kb_dict[doc] == "positive" else "bad"
                context += f"Review: {doc[:300]}...\nSentiment: {doc_map}.\n\n"
            short_text = text[:1000]
            prompt = f"{context}Review: {short_text}\nSentiment: "
            output = self.generate(prompt, max_new_tokens=5)
            pred = "positive" if ("good" in output or "positive" in output) else \
                   "negative" if ("bad" in output or "negative" in output) else "unknown"
            results.append({"method": "cloze_rag", "pred": pred, "true": label})
        return results


def main():
    class Args:
        pass
    args = Args()
    args.model_name = 'Qwen/Qwen3-0.6B'
    args.output_dir = 'IMDB_qwen3_0.6b_benchmark'
    
    
    loader = IMDBDataLoader(n_kb=2000, n_eval=100)
    kb_data, eval_data = loader.load()
    kb_dict = {text: label for text, label in kb_data}
    
    retriever = FAISSRetriever()
    retriever.build([text for text, _ in kb_data])
    
    classifier = QwenPromptClassifier(model_name=args.model_name)
    
    all_results = []
    
    all_results.extend(classifier.run_cloze_simple(eval_data))
    all_results.extend(classifier.run_pet_style(eval_data))
    all_results.extend(classifier.run_cloze_rag(eval_data, retriever, kb_dict))
    
    all_results.extend(classifier.run_prefix_zeroshot(eval_data))
    all_results.extend(classifier.run_prefix_rag(eval_data, retriever, kb_dict))
    all_results.extend(classifier.run_prefix_cot(eval_data))
    all_results.extend(classifier.run_decompose_rag(eval_data, retriever, kb_dict))
    
    # Process and save results
    out_df = pd.DataFrame(all_results)
    out_df["correct"] = out_df["pred"] == out_df["true"]
    
    print(f"\n--- IMDB Benchmark Results ({args.model_name}) ---")
    for method in out_df["method"].unique():
        sub = out_df[out_df["method"] == method]
        acc = accuracy_score(sub["true"], sub["pred"])
        f1 = f1_score(sub["true"], sub["pred"], average="weighted")
        print(f"[{method}] Acc: {acc:.4f} | F1: {f1:.4f}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results_imdb.csv")
    out_df.to_csv(out_file, index=False)
    print(f"\nSaved detailed results to: {out_file}")

if __name__ == "__main__":
    main()
