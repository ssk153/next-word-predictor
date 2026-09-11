import argparse
import torch
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from tqdm import tqdm
from utils import (
    read_dev_data,
    save_predictions,
    compute_accuracy,
    load_model
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def predict_fourgram_topk(context: str, candidates: list, 
                          unigram_counts: Counter, bigram_counts: Counter, 
                          trigram_counts: Counter, fourgram_counts: Counter, 
                          total_unigrams: int, top_k: int = 100) -> list:
    if not candidates:
        return []
        
    tokens = str(context).strip().split()
    w_prev = tokens[-1] if len(tokens) >= 1 else None
    w_prev2 = tokens[-2] if len(tokens) >= 2 else None
    w_prev3 = tokens[-3] if len(tokens) >= 3 else None
    
    c_fourgram = trigram_counts.get((w_prev3, w_prev2, w_prev), 0) if w_prev3 and w_prev2 and w_prev else 0
    c_trigram = bigram_counts.get((w_prev2, w_prev), 0) if w_prev2 and w_prev else 0
    c_bigram = unigram_counts.get(w_prev, 0) if w_prev else 0
    
    scored_candidates = []
    
    # cand_tuple is (word, first_token_id, token_ids)
    for cand_tuple in candidates:
        cand = cand_tuple[0]
        prob = 0.0
        
        # 1. Fourgram probability
        if c_fourgram > 0:
            prob = fourgram_counts.get((w_prev3, w_prev2, w_prev, cand), 0) / c_fourgram
                
        # 2. Backoff to trigram
        if prob == 0.0 and c_trigram > 0:
            prob = trigram_counts.get((w_prev2, w_prev, cand), 0) / c_trigram
        
        # 3. Backoff to bigram
        if prob == 0.0 and c_bigram > 0:
            prob = bigram_counts.get((w_prev, cand), 0) / c_bigram
                
        # 4. Backoff to unigram
        if prob == 0.0:
            prob = unigram_counts.get(cand, 0) / total_unigrams if total_unigrams > 0 else 0.0
            
        scored_candidates.append((prob, cand_tuple))
            
    # Sort by probability, descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_candidates[:top_k]]

def predict_batch_hybrid(model, tokenizer, contexts: List[str], first_letters: List[str], topk_candidates_per_context: List[List[Tuple]]) -> List[str]:
    tokenizer.padding_side = 'left'
    inputs = tokenizer(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
    logprobs_cpu = next_token_logprobs.cpu()
    
    multi_token_evals = []
    context_top_cands = []
    
    for b in range(len(contexts)):
        candidates = topk_candidates_per_context[b]
        
        if not candidates:
            context_top_cands.append([])
            continue
            
        # STAGE 1: Rank by first token score (Qwen score)
        scored_cands = []
        for word, first_token_id, token_ids in candidates:
            score = logprobs_cpu[b, first_token_id].item()
            scored_cands.append((score, word, token_ids))
            
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        # Take the best candidates based on Qwen's first-token approximation
        # (Fourgram already narrowed it down to `top_k`, now we rank by Qwen)
        top_20 = scored_cands[:20]
        context_top_cands.append(top_20)
        
        for score, word, token_ids in top_20:
            if len(token_ids) > 1:
                multi_token_evals.append((b, word, token_ids, score))
                
    multi_token_final_scores = {}
    
    if multi_token_evals:
        batch_input_ids = []
        for b, word, token_ids, first_score in multi_token_evals:
            ctx_ids = inputs.input_ids[b].tolist()
            pad_id = tokenizer.pad_token_id
            non_pad_idx = 0
            while non_pad_idx < len(ctx_ids) and ctx_ids[non_pad_idx] == pad_id:
                non_pad_idx += 1
            clean_ctx_ids = ctx_ids[non_pad_idx:]
            
            full_ids = clean_ctx_ids + token_ids[:-1]
            batch_input_ids.append(full_ids)
            
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_len - len(ids)
            padded_input_ids.append([tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))
            
        input_tensor = torch.tensor(padded_input_ids, device=model.device)
        mask_tensor = torch.tensor(attention_masks, device=model.device)
        
        eval_logits_list = []
        mini_batch_size = 4
        
        for i in range(0, len(input_tensor), mini_batch_size):
            batch_in = input_tensor[i:i+mini_batch_size]
            batch_mask = mask_tensor[i:i+mini_batch_size]
            with torch.no_grad():
                eval_outputs = model(input_ids=batch_in, attention_mask=batch_mask, use_cache=False)
                eval_logits_list.append(eval_outputs.logits.cpu())
                
        eval_logits = torch.cat(eval_logits_list, dim=0)
            
        for i, (b, word, token_ids, first_score) in enumerate(multi_token_evals):
            seq_len = len(token_ids) - 1
            targets = token_ids[1:]
            
            cand_logits = eval_logits[i, -seq_len:, :]
            cand_logprobs = torch.nn.functional.log_softmax(cand_logits, dim=-1)
            
            score = first_score
            for t in range(seq_len):
                score += cand_logprobs[t, targets[t]].item()
                
            multi_token_final_scores[(b, word)] = score

    predictions = []
    for b in range(len(contexts)):
        top_20 = context_top_cands[b]
        if not top_20:
            predictions.append("")
            continue
            
        best_cand = ""
        best_score = -float('inf')
        
        for first_score, word, token_ids in top_20:
            if len(token_ids) == 1:
                score = first_score
            else:
                score = multi_token_final_scores[(b, word)]
                
            if score > best_score:
                best_score = score
                best_cand = word
                
        predictions.append(best_cand)
        
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Hybrid Fourgram + Qwen LM for Next Word Prediction")
    parser.add_argument("--dev", required=True, help="Path to dev/test CSV file")
    parser.add_argument("--output", default="predictions_hybrid.csv", help="Output predictions CSV file")
    parser.add_argument("--fourgram_model_path", default="fourgram_model.pkl", help="Path to trained fourgram model")
    parser.add_argument("--qwen_vocab_path", default="qwen_filtered_vocab.pkl", help="Path to Qwen precomputed candidates")
    parser.add_argument("--device", default="auto", help="Device to run on (e.g., 'cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev rows for quick testing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Qwen model")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top candidates from Fourgram to pass to Qwen")
    args = parser.parse_args()
    
    print(f"Loading Fourgram model from {args.fourgram_model_path}...")
    loaded_4g = load_model(args.fourgram_model_path)
    if not loaded_4g:
        print("Error: Fourgram model not found. Please train it first.")
        return
    vocab_4g, unigrams, bigrams, trigrams, fourgrams, total_unigrams = loaded_4g
    print("Fourgram model loaded successfully.")

    print(f"Loading Qwen vocabulary/candidates from {args.qwen_vocab_path}...")
    candidates_by_letter = load_model(args.qwen_vocab_path)
    if not candidates_by_letter:
        print("Error: Qwen vocabulary not found. Please generate it first using qwen_model.py.")
        return
    print("Qwen candidates loaded.")

    print(f"Loading pretrained model tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading pretrained model: {MODEL_NAME}...")
    device_map = args.device if args.device != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map=device_map
    )
    model.eval()
    
    print(f"Reading dev data from {args.dev}...")
    dev_df = read_dev_data(args.dev, limit=args.limit)
    
    contexts = [str(c).strip() for c in dev_df['context'].values]
    first_letters = [str(l) for l in dev_df['first letter'].values]
    
    predictions = []
    print(f"Predicting in batches of {args.batch_size}...")
    
    for i in tqdm(range(0, len(contexts), args.batch_size), desc="Hybrid Predicting"):
        batch_contexts = contexts[i:i + args.batch_size]
        batch_letters = first_letters[i:i + args.batch_size]
        
        batch_topk_candidates = []
        for ctx, fl in zip(batch_contexts, batch_letters):
            fl_lower = fl.lower()
            all_cands_for_letter = candidates_by_letter.get(fl_lower, [])
            
            if not all_cands_for_letter:
                batch_topk_candidates.append([])
                continue
                
            topk_cands = predict_fourgram_topk(
                context=ctx,
                candidates=all_cands_for_letter,
                unigram_counts=unigrams,
                bigram_counts=bigrams,
                trigram_counts=trigrams,
                fourgram_counts=fourgrams,
                total_unigrams=total_unigrams,
                top_k=args.top_k
            )
            batch_topk_candidates.append(topk_cands)
            
        batch_preds = predict_batch_hybrid(model, tokenizer, batch_contexts, batch_letters, batch_topk_candidates)
        predictions.extend(batch_preds)
        
    save_predictions(dev_df, predictions, args.output)
    print(f"Predictions saved to {args.output}")
    
    if 'answer' in dev_df.columns:
        acc, correct, total = compute_accuracy(predictions, dev_df)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()

