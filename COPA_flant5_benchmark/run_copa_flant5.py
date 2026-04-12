"""
Unified COPA Benchmark Script for BERT, FlanT5, and Qwen
"""
import argparse
import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset


class COPADataLoader:
    def __init__(self, n_eval=100):
        self.n_eval = n_eval

    def load(self):
        print(f"Loading COPA dataset (Eval: {self.n_eval})...")
        ds = load_dataset("super_glue", "copa")
        
        kb_data = []
        for item in ds["train"]:
            kb_data.append({
                "premise": item["premise"], 
                "c1": item["choice1"], 
                "c2": item["choice2"], 
                "q": item["question"],
                "answer": "1" if item["label"] == 0 else "2"
            })
            
        eval_data = []
        val_split = list(ds["validation"])[:self.n_eval]
        for item in val_split:
            eval_data.append({
                "premise": item["premise"], 
                "c1": item["choice1"], 
                "c2": item["choice2"], 
                "q": item["question"],
                "answer": "1" if item["label"] == 0 else "2"
            })
            
        return kb_data, eval_data


class FAISSRetriever:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model_name
        self.model = None
        self.index = None
        self.kb = []

    def build(self, kb_data):
        print(f"Building FAISS index for {len(kb_data)} documents...")
        self.model = SentenceTransformer(self.embed_model_name)
        texts = [doc["premise"] for doc in kb_data]
        embeddings = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.kb = kb_data
        return self

    def retrieve(self, query, top_k=2):
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        return [self.kb[i] for i in ids[0]]


def match_answer(pred, true_ans):
    pred = str(pred).lower()
    true_ans = str(true_ans)
    if true_ans in pred: return True
    if true_ans == "1" and ("first" in pred or "one" in pred): return True
    if true_ans == "2" and ("second" in pred or "two" in pred): return True
    return False


class UnifiedCOPAPrompter:
    def __init__(self, model_name, model_type, device=None):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} ({model_type}) on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_type == "bert":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        elif model_type == "flant5":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        elif model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    def process_prompt(self, prompt, target_words=["1", "2"], max_new_tokens=5):
        if self.model_type == "bert":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            mask_idx = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_idx[0]) == 0: return "unknown"
            with torch.no_grad():
                logits = self.model(**inputs).logits
            mask_logits = logits[0, mask_idx[1][0], :]
            probs = torch.nn.functional.softmax(mask_logits, dim=-1)
            
            scores = {}
            for w in target_words:
                w_id = self.tokenizer.encode(w, add_special_tokens=False)
                if len(w_id) > 0:
                    scores[w] = probs[w_id[0]].item()
                else:
                    scores[w] = 0
            best_id_word = max(scores, key=scores.get)
            return best_id_word
        
        elif self.model_type == "flant5":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
        elif self.model_type == "causal":
            if "pad_token" not in self.tokenizer.special_tokens_map:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    def _get_target_string(self):
        if self.model_type == "bert": return "[MASK]"
        if self.model_type == "flant5": return "<extra_id_0>"
        return ""

    def run_all(self, eval_data, retriever):
        all_results = []
        for method in ["cloze_simple", "pet_style", "cloze_rag", "prefix_zeroshot", "prefix_rag", "prefix_cot", "decompose_rag"]:
            for item in tqdm(eval_data, desc=method):
                p = item["premise"]
                c1 = item["c1"]; c2 = item["c2"]; q = item["q"]
                true_ans = item["answer"]
                target = self._get_target_string()
                
                connector = "because" if q == "cause" else "therefore"
                
                if method == "cloze_simple":
                    prompt = f"Premise: {p}\nChoice 1: {c1}\nChoice 2: {c2}\nThe correct choice is Choice {target}"
                    pred = self.process_prompt(prompt)
                
                elif method == "pet_style":
                    prompt = f"{p} {connector} {target} ."
                    # Since target is token 1 or 2, tell it what it means
                    prompt = f"1: {c1}. 2: {c2}. {p} {connector} choice {target}."
                    pred = self.process_prompt(prompt)
                    
                elif method == "cloze_rag":
                    shots = retriever.retrieve(p, top_k=1)
                    shot_str = f"P: {shots[0]['premise']}\nC1: {shots[0]['c1']}\nC2: {shots[0]['c2']}\nAns: Choice {shots[0]['answer']}.\n\n" if shots else ""
                    prompt = f"{shot_str}P: {p}\nC1: {c1}\nC2: {c2}\nAns: Choice {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_zeroshot":
                    prompt = f"Determine the {q} of the premise.\nPremise: {p}\n1) {c1}\n2) {c2}\nAnswer (1 or 2): {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_rag":
                    shots = retriever.retrieve(p, top_k=2)
                    shot_str = "".join([f"Premise: {s['premise']}\n1) {s['c1']}\n2) {s['c2']}\nQ: {s['q']}\nCorrect: {s['answer']}\n\n" for s in shots])
                    prompt = f"{shot_str}Premise: {p}\n1) {c1}\n2) {c2}\nQ: {q}\nCorrect: {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_cot":
                    prompt = f"Premise: {p}\n1) {c1}\n2) {c2}\nLet's think step by step to find the {q}. {target}"
                    pred = self.process_prompt(prompt, max_new_tokens=40)
                    
                elif method == "decompose_rag":
                    sub_prompt = f"Premise: {p}\nQuestion: What is the main subject? {target}"
                    sub_ans = self.process_prompt(sub_prompt, max_new_tokens=10)
                    
                    prompt2 = f"Subject: {sub_ans}\nPremise: {p}\nQuestion: What is its {q}?\n1) {c1}\n2) {c2}\nChoice: {target}"
                    pred = self.process_prompt(prompt2)

                all_results.append({"method": method, "pred": pred, "true": true_ans})
        return all_results

def main():
    class Args:
        pass
    args = Args()
    args.model_name = 'google/flan-t5-base'
    args.model_type = 'flant5'
    args.output_dir = 'COPA_flant5_benchmark'
    
    
    loader = COPADataLoader(n_eval=100) # COPA validation set has 100 items
    kb_data, eval_data = loader.load()
    
    retriever = FAISSRetriever()
    retriever.build(kb_data)
    
    prompter = UnifiedCOPAPrompter(args.model_name, args.model_type)
    results = prompter.run_all(eval_data, retriever)
    
    out_df = pd.DataFrame(results)
    out_df["correct"] = out_df.apply(lambda r: match_answer(r["pred"], r["true"]), axis=1)
    
    print(f"\n--- COPA Benchmark Results ({args.model_name}) ---")
    for method in out_df["method"].unique():
        sub = out_df[out_df["method"] == method]
        acc = sub["correct"].mean()
        print(f"[{method}] Acc: {acc:.4f}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results_copa.csv")
    out_df.to_csv(out_file, index=False)

if __name__ == "__main__":
    main()
