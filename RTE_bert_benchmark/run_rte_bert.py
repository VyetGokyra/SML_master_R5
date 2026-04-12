"""
Unified RTE Benchmark Script for BERT, FlanT5, and Qwen
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


class RTEDataLoader:
    def __init__(self, n_eval=100):
        self.n_eval = n_eval

    def load(self):
        print(f"Loading RTE dataset (Eval: {self.n_eval})...")
        ds = load_dataset("super_glue", "rte")
        
        kb_data = []
        for item in ds["train"]:
            kb_data.append({
                "premise": item["premise"], 
                "hypothesis": item["hypothesis"], 
                "answer": "yes" if item["label"] == 0 else "no"
            })
            
        eval_data = []
        val_split = list(ds["validation"])[:self.n_eval]
        for item in val_split:
            eval_data.append({
                "premise": item["premise"], 
                "hypothesis": item["hypothesis"], 
                "answer": "yes" if item["label"] == 0 else "no"
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
        texts = [doc["premise"] + " " + doc["hypothesis"] for doc in kb_data]
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
    true_ans = str(true_ans).lower()
    # Handle causal models that might output "Answer: yes"
    if true_ans == "yes":
        if "yes" in pred or "true" in pred: return True
        return False
    else:
        if "no" in pred or "false" in pred: return True
        return False


class UnifiedRTEPrompter:
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

    def process_prompt(self, prompt, target_words=["yes", "no"], max_new_tokens=10):
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
            return max(scores, key=scores.get)
        
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
                p = item["premise"][:500]
                h = item["hypothesis"][:200]
                true_ans = item["answer"]
                target = self._get_target_string()
                
                if method == "cloze_simple":
                    prompt = f"Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis? {target}"
                    pred = self.process_prompt(prompt)
                
                elif method == "pet_style":
                    prompt = f"{p} ? {target} , {h}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "cloze_rag":
                    shots = retriever.retrieve(p + " " + h, top_k=1)
                    shot_str = f"P: {shots[0]['premise']}\nH: {shots[0]['hypothesis']}\nAns: {shots[0]['answer']}.\n" if shots else ""
                    prompt = f"{shot_str}P: {p}\nH: {h}\nAns: {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_zeroshot":
                    prompt = f"Given the premise: \"{p}\", does it imply \"{h}\"? Answer yes or no. Answer: {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_rag":
                    shots = retriever.retrieve(p + " " + h, top_k=2)
                    shot_str = "".join([f"Premise: {s['premise']}\nHypothesis: {s['hypothesis']}\nDoes it entail? {s['answer']}\n" for s in shots])
                    prompt = f"{shot_str}Premise: {p}\nHypothesis: {h}\nDoes it entail? {target}"
                    pred = self.process_prompt(prompt)
                    
                elif method == "prefix_cot":
                    prompt = f"Premise: {p}\nHypothesis: {h}\nLet's think step by step to see if they mean the same thing. {target}"
                    pred = self.process_prompt(prompt, max_new_tokens=40)
                    
                elif method == "decompose_rag":
                    # For RTE Decompose: Ask what is the core entity/action in hypothesis
                    sub_prompt = f"Hypothesis: {h}\nQuestion: What is the main action? {target}"
                    sub_ans = self.process_prompt(sub_prompt, max_new_tokens=15)
                    
                    prompt2 = f"Main action: {sub_ans}\nPremise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis? {target}"
                    pred = self.process_prompt(prompt2)

                all_results.append({"method": method, "pred": pred, "true": true_ans})
        return all_results

def main():
    class Args:
        pass
    args = Args()
    args.model_name = 'bert-base-uncased'
    args.model_type = 'bert'
    args.output_dir = 'RTE_bert_benchmark'
    
    
    loader = RTEDataLoader(n_eval=100) # Testing on 100
    kb_data, eval_data = loader.load()
    
    retriever = FAISSRetriever()
    retriever.build(kb_data)
    
    prompter = UnifiedRTEPrompter(args.model_name, args.model_type)
    results = prompter.run_all(eval_data, retriever)
    
    out_df = pd.DataFrame(results)
    out_df["correct"] = out_df.apply(lambda r: match_answer(r["pred"], r["true"]), axis=1)
    
    print(f"\n--- RTE Benchmark Results ({args.model_name}) ---")
    for method in out_df["method"].unique():
        sub = out_df[out_df["method"] == method]
        acc = sub["correct"].mean()
        print(f"[{method}] Acc: {acc:.4f}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "results_rte.csv")
    out_df.to_csv(out_file, index=False)

if __name__ == "__main__":
    main()
