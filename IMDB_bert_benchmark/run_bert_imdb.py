"""
IMDB Benchmark for BERT (Masked Language Modeling)
"""
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
        
        # Consistent sampling
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
        if not self.index:
            raise ValueError("Index not built yet.")
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        return [(self.docs[i], float(scores[0][j])) for j, i in enumerate(ids[0])]


class BERTPromptClassifier:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _predict(self, prompt, target_words):
        """
        Evaluate P([MASK] = word) for each word in target_words.
        Returns the word with the highest logit.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Find [MASK] token
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        
        if len(mask_token_index) == 0:
            # Fallback if text is too long and MASK gets truncated
            return "unknown"
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        mask_token_logits = logits[0, mask_token_index, :]
        mask_token_probs = F.softmax(mask_token_logits, dim=-1)[0]
        
        scores = {}
        target_ids = {word: self.tokenizer.encode(word, add_special_tokens=False) for word in target_words}
        
        for word, ids in target_ids.items():
            # If word is broken into subwords, we take the average logit or the first token.
            # For simplicity, assuming single token words like "good", "bad", "positive", "negative"
            if len(ids) > 0:
                scores[word] = mask_token_probs[ids[0]].item()
            else:
                scores[word] = 0.0

        best_word = max(scores, key=scores.get)
        return best_word

    def run_cloze_simple(self, eval_data):
        """
        [review] Because it was [MASK]. 
        Answer space: {good/great vs bad/terrible}
        Based on NguyenNamHoang's manual cloze.
        """
        results = []
        pos_words = ["good"]
        neg_words = ["bad"]
        
        for text, label in tqdm(eval_data, desc="Cloze Simple"):
            # Truncate text to leave room for prompt
            short_text = text[:1500] 
            prompt = f"{short_text}\nBecause it was [MASK]."
            best_word = self._predict(prompt, pos_words + neg_words)
            pred = "positive" if best_word in pos_words else "negative"
            results.append({"method": "cloze_simple", "pred": pred, "true": label})
        return results

    def run_pet_style(self, eval_data):
        """
        [review] All in all, the movie was [MASK].
        """
        results = []
        target_words = ["good", "bad"] # simple mapping
        for text, label in tqdm(eval_data, desc="PET-style"):
            short_text = text[:1500]
            prompt = f"{short_text}\nAll in all, the movie was [MASK]."
            best_word = self._predict(prompt, target_words)
            pred = "positive" if best_word == "good" else "negative"
            results.append({"method": "pet_style", "pred": pred, "true": label})
        return results

    def run_cloze_rag(self, eval_data, retriever, kb_dict):
        """
        RAG + Cloze:
        Review: [example_review] Sentiment: [label_pos_or_neg]
        Review: [review] Sentiment: [MASK]
        """
        results = []
        # For BERT mask prediction, mapping to single tokens is safer.
        target_words = ["good", "bad"] 
        
        for text, label in tqdm(eval_data, desc="Cloze RAG"):
            retrieved = retriever.retrieve(text, top_k=2)
            context = ""
            for doc, _ in retrieved:
                doc_label = kb_dict[doc]
                doc_map = "good" if doc_label == "positive" else "bad"
                context += f"Review: {doc[:300]}...\nSentiment: {doc_map}.\n\n"
                
            short_text = text[:1000]
            prompt = f"{context}Review: {short_text}\nSentiment: [MASK]."
            best_word = self._predict(prompt, target_words)
            pred = "positive" if best_word == "good" else "negative"
            results.append({"method": "cloze_rag", "pred": pred, "true": label})
        return results

    def run_prefix_zeroshot(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix Zero-shot"):
            prompt = f"Review: {text[:1500]}\nSentiment: [MASK]."
            best_word = self._predict(prompt, ["good", "bad"])
            pred = "positive" if best_word == "good" else "negative"
            results.append({"method": "prefix_zeroshot", "pred": pred, "true": label})
        return results

    def run_prefix_rag(self, eval_data, retriever, kb_dict):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix RAG"):
            retrieved = retriever.retrieve(text, top_k=2)
            context = ""
            for doc, _ in retrieved:
                doc_map = "good" if kb_dict[doc] == "positive" else "bad"
                context += f"Review: {doc[:300]}\nSentiment: {doc_map}.\n"
            prompt = f"Context:\n{context}\nReview: {text[:1000]}\nSentiment: [MASK]."
            best_word = self._predict(prompt, ["good", "bad"])
            pred = "positive" if best_word == "good" else "negative"
            results.append({"method": "prefix_rag", "pred": pred, "true": label})
        return results

    def run_prefix_cot(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Prefix CoT"):
            prompt = f"Review: {text[:1000]}\nLet's think step by step. The review contains positive and negative points. Overall, it is [MASK]."
            best_word = self._predict(prompt, ["good", "bad"])
            pred = "positive" if best_word == "good" else "negative"
            results.append({"method": "prefix_cot", "pred": pred, "true": label})
        return results

    def run_decompose_rag(self, eval_data):
        results = []
        for text, label in tqdm(eval_data, desc="Decompose RAG"):
            aspects = ["acting", "plot", "directing"]
            votes = {"good": 0, "bad": 0}
            for aspect in aspects:
                prompt = f"Review: {text[:1000]}\nQuestion: How is the {aspect}? The {aspect} is [MASK]."
                best_word = self._predict(prompt, ["good", "bad"])
                votes[best_word] += 1
            pred = "positive" if votes["good"] >= votes["bad"] else "negative"
            results.append({"method": "decompose_rag", "pred": pred, "true": label})
        return results

def main():
    loader = IMDBDataLoader(n_kb=2000, n_eval=100)
    kb_data, eval_data = loader.load()
    kb_dict = {text: label for text, label in kb_data}
    
    retriever = FAISSRetriever()
    retriever.build([text for text, _ in kb_data])
    
    classifier = BERTPromptClassifier(model_name="bert-base-uncased")
    
    all_results = []
    
    # Run evaluated conditions
    res_simple = classifier.run_cloze_simple(eval_data)
    all_results.extend(res_simple)
    
    res_pet = classifier.run_pet_style(eval_data)
    all_results.extend(res_pet)
    
    res_rag = classifier.run_cloze_rag(eval_data, retriever, kb_dict)
    all_results.extend(res_rag)
    
    res_pz = classifier.run_prefix_zeroshot(eval_data)
    all_results.extend(res_pz)
    
    res_pr = classifier.run_prefix_rag(eval_data, retriever, kb_dict)
    all_results.extend(res_pr)
    
    res_pc = classifier.run_prefix_cot(eval_data)
    all_results.extend(res_pc)
    
    res_dr = classifier.run_decompose_rag(eval_data)
    all_results.extend(res_dr)
    
    # Process and save results
    out_df = pd.DataFrame(all_results)
    out_df["correct"] = out_df["pred"] == out_df["true"]
    
    print("\n--- IMDB Benchmark Results (BERT) ---")
    for method in out_df["method"].unique():
        sub = out_df[out_df["method"] == method]
        acc = accuracy_score(sub["true"], sub["pred"])
        f1 = f1_score(sub["true"], sub["pred"], pos_label="positive")
        print(f"[{method}] Acc: {acc:.4f} | F1: {f1:.4f}")
        
    out_file = "/home/tienpv16/Documents/Research/project_SML/SML_master_R5/IMDB_bert_benchmark/results_imdb.csv"
    out_df.to_csv(out_file, index=False)
    print(f"\nSaved detailed results to: {out_file}")


if __name__ == "__main__":
    main()
