import json
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge_score...")
    import subprocess
    subprocess.run(["pip", "install", "rouge_score"])
    from rouge_score import rouge_scorer

from .config import CHECKPOINT_VI, CHECKPOINT_EN, VAL_PATH
from summarization.vit5_wrapper import VIT5Summarizer

def evaluate(lang="vi"):
    checkpoint = CHECKPOINT_VI if lang == "vi" else CHECKPOINT_EN
    text_key = "text_vi" if lang == "vi" else "text_en"
    summary_key = "summary_vi" if lang == "vi" else "summary_en"
    
    print(f"Evaluating {lang} using checkpoint: {checkpoint}")
    summarizer = VIT5Summarizer(checkpoint)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    with open(VAL_PATH, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for item in tqdm(data[:100], desc=f"Evaluating {lang}"): # Giới hạn 100 mẫu để test nhanh
        src_text = item[text_key]
        ref_summary = item[summary_key]
        
        gen_res = summarizer.summarize(src_text)
        gen_summary = gen_res["summary"]
        
        scores = scorer.score(ref_summary, gen_summary)
        for k in all_scores:
            all_scores[k].append(scores[k].fmeasure)
            
    avg_scores = {k: sum(v)/len(v) for k, v in all_scores.items() if v}
    print(f"Results for {lang}:")
    for k, v in avg_scores.items():
        print(f"  {k}: {v:.4f}")
        
    return avg_scores

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["vi", "en", "both"], default="vi")
    args = p.parse_args()
    
    if args.lang in ["vi", "both"]:
        evaluate("vi")
    if args.lang in ["en", "both"]:
        evaluate("en")
