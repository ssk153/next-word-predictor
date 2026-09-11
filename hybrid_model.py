import argparse
import math
import torch
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from tqdm import tqdm
from utils import (
    read_dev_data,
    save_predictions,
    compute_accuracy,
    load_model
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def smoothed_fourgram_topk(context: str, candidates: list, 
                           unigrams: Counter, bigrams: Counter, 
                           trigrams: Counter, fourgrams: Counter,
                           U_1: Counter, U_2: Counter, U_3: Counter,
                           total_unigrams: int, D: float,
                           top_k: int = 100) -> List[Tuple]:
    """
    Score candidates using Interpolated Absolute Discounting (Option 3),
    then return the top_k candidates WITH their smoothed fourgram probabilities.
    
    Returns: list of (cand_tuple, fourgram_prob) where cand_tuple is (word, first_token_id, token_ids)
    """
    if not candidates:
        return []
        
    tokens = str(context).strip().split()
    w_prev = tokens[-1] if len(tokens) >= 1 else None
    w_prev2 = tokens[-2] if len(tokens) >= 2 else None
    w_prev3 = tokens[-3] if len(tokens) >= 3 else None
    
    c_3 = trigrams.get((w_prev3, w_prev2, w_prev), 0) if w_prev3 and w_prev2 and w_prev else 0
    c_2 = bigrams.get((w_prev2, w_prev), 0) if w_prev2 and w_prev else 0
    c_1 = unigrams.get(w_prev, 0) if w_prev else 0
    
    u_3 = U_3.get((w_prev3, w_prev2, w_prev), 0) if w_prev3 and w_prev2 and w_prev else 0
    u_2 = U_2.get((w_prev2, w_prev), 0) if w_prev2 and w_prev else 0
    u_1 = U_1.get(w_prev, 0) if w_prev else 0
    
    lambda_4 = (D / c_3 * u_3) if c_3 > 0 else 1.0
    lambda_3 = (D / c_2 * u_2) if c_2 > 0 else 1.0
    lambda_2 = (D / c_1 * u_1) if c_1 > 0 else 1.0
    
    scored_candidates = []
    
    # cand_tuple is (word, first_token_id, token_ids)
    for cand_tuple in candidates:
        cand = cand_tuple[0]
        
        # P1 (Unigram)
        p_1 = unigrams.get(cand, 0) / total_unigrams if total_unigrams > 0 else 0.0
        
        # P2 (Bigram with Absolute Discounting)
        if c_1 > 0:
            p_2 = max(bigrams.get((w_prev, cand), 0) - D, 0) / c_1 + lambda_2 * p_1
        else:
            p_2 = p_1
            
        # P3 (Trigram with Absolute Discounting)
        if c_2 > 0:
            p_3 = max(trigrams.get((w_prev2, w_prev, cand), 0) - D, 0) / c_2 + lambda_3 * p_2
        else:
            p_3 = p_2
            
        # P4 (Fourgram with Absolute Discounting)
        if c_3 > 0:
            p_4 = max(fourgrams.get((w_prev3, w_prev2, w_prev, cand), 0) - D, 0) / c_3 + lambda_4 * p_3
        else:
            p_4 = p_3
            
        scored_candidates.append((p_4, cand_tuple))
            
    # Sort by smoothed probability, descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    # Return (cand_tuple, fourgram_prob) pairs
    return [(c[1], c[0]) for c in scored_candidates[:top_k]]


def predict_batch_hybrid(model, tokenizer, contexts: List[str], first_letters: List[str], 
                         topk_candidates_per_context: List[List[Tuple]], alpha: float) -> List[str]:
    """
    Hybrid prediction: combine Smoothed Fourgram probabilities with Qwen log-probs.
    
    final_score = alpha * log(P_fourgram + 1e-12) + (1 - alpha) * qwen_logprob
    """
    tokenizer.padding_side = 'left'
    inputs = tokenizer(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
    logprobs_cpu = next_token_logprobs.cpu()
    
    multi_token_evals = []
    # Store per-context: list of (word, token_ids, first_token_qwen_score, fourgram_prob)
    context_all_cands = []
    
    for b in range(len(contexts)):
        candidates_with_probs = topk_candidates_per_context[b]
        
        if not candidates_with_probs:
            context_all_cands.append([])
            continue
            
        cand_info = []
        for (word, first_token_id, token_ids), fourgram_prob in candidates_with_probs:
            qwen_first_score = logprobs_cpu[b, first_token_id].item()
            cand_info.append((word, token_ids, qwen_first_score, fourgram_prob))
            
            if len(token_ids) > 1:
                multi_token_evals.append((b, word, token_ids, qwen_first_score))
        
        context_all_cands.append(cand_info)
                
    # Stage 2: Full-word probability for multi-token candidates
    multi_token_final_scores: Dict[Tuple, float] = {}
    
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

    # Final selection: combine fourgram prob + qwen logprob
    predictions = []
    for b in range(len(contexts)):
        cand_info = context_all_cands[b]
        if not cand_info:
            predictions.append("")
            continue
            
        best_cand = ""
        best_combined = -float('inf')
        
        for word, token_ids, qwen_first_score, fourgram_prob in cand_info:
            # Get full Qwen log-probability
            if len(token_ids) == 1:
                qwen_logprob = qwen_first_score
            else:
                qwen_logprob = multi_token_final_scores[(b, word)]
            
            # Combined score: alpha * log(P_fourgram + eps) + (1 - alpha) * qwen_logprob
            fourgram_logscore = math.log(fourgram_prob + 1e-12)
            combined = alpha * fourgram_logscore + (1.0 - alpha) * qwen_logprob
                
            if combined > best_combined:
                best_combined = combined
                best_cand = word
                
        predictions.append(best_cand)
        
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Hybrid Smoothed-Fourgram + Qwen LM for Next Word Prediction")
    parser.add_argument("--dev", required=True, help="Path to dev/test CSV file")
    parser.add_argument("--output", default="predictions_hybrid.csv", help="Output predictions CSV file")
    parser.add_argument("--fourgram_model_path", default="fourgram_model.pkl", help="Path to trained fourgram model")
    parser.add_argument("--qwen_vocab_path", default="qwen_filtered_vocab.pkl", help="Path to Qwen precomputed candidates")
    parser.add_argument("--device", default="auto", help="Device to run on (e.g., 'cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev rows for quick testing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Qwen model")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top candidates from Smoothed Fourgram to pass to Qwen")
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for Smoothed Fourgram score (0-1). Higher = more fourgram influence.")
    parser.add_argument("--discount", type=float, default=0.75, help="Discount factor D for Absolute Discounting smoothing")
    parser.add_argument("--eval_candidates", action="store_true", 
                        help="Evaluate candidate generation recall@k without invoking Qwen.")
    args = parser.parse_args()
    
    # Load fourgram model
    print(f"Loading Fourgram model from {args.fourgram_model_path}...")
    loaded_4g = load_model(args.fourgram_model_path)
    if not loaded_4g:
        print("Error: Fourgram model not found. Please train it first.")
        return
    vocab_4g, unigrams, bigrams, trigrams, fourgrams, total_unigrams = loaded_4g
    print("Fourgram model loaded successfully.")
    
    # Precompute unique continuations for Absolute Discounting
    print("Precomputing unique continuations for Absolute Discounting smoothing...")
    U_3 = Counter()
    for (w1, w2, w3, w4) in fourgrams.keys():
        U_3[(w1, w2, w3)] += 1
        
    U_2 = Counter()
    for (w1, w2, w3) in trigrams.keys():
        U_2[(w1, w2)] += 1
        
    U_1 = Counter()
    for (w1, w2) in bigrams.keys():
        U_1[w1] += 1

    # Load Qwen candidates (needed for both modes — contains tokenizer metadata)
    print(f"Loading Qwen vocabulary/candidates from {args.qwen_vocab_path}...")
    candidates_by_letter = load_model(args.qwen_vocab_path)
    if not candidates_by_letter:
        print("Error: Qwen vocabulary not found. Please generate it first using qwen_model.py.")
        return
    print("Qwen candidates loaded.")

    print(f"Reading dev data from {args.dev}...")
    dev_df = read_dev_data(args.dev, limit=args.limit)
    
    contexts = [str(c).strip() for c in dev_df['context'].values]
    first_letters = [str(l) for l in dev_df['first letter'].values]

    # ─── Candidate Evaluation Mode (no Qwen) ───
    if args.eval_candidates:
        if 'answer' not in dev_df.columns:
            print("Error: --eval_candidates requires a dev set with an 'answer' column.")
            return
        
        answers = [str(a) for a in dev_df['answer'].values]
        k_values = [1, 3, 5, 10, 20, 50, 100]
        recall_counts = {k: 0 for k in k_values}
        total = len(contexts)
        
        print(f"Evaluating candidate recall@k (top_k={args.top_k}, discount={args.discount})...")
        
        for ctx, fl, answer in tqdm(zip(contexts, first_letters, answers), total=total, desc="Eval Candidates"):
            fl_lower = fl.lower()
            all_cands_for_letter = candidates_by_letter.get(fl_lower, []) # type: ignore
            
            if not all_cands_for_letter:
                continue
                
            topk_with_probs = smoothed_fourgram_topk(
                context=ctx,
                candidates=all_cands_for_letter,
                unigrams=unigrams,
                bigrams=bigrams,
                trigrams=trigrams,
                fourgrams=fourgrams,
                U_1=U_1, U_2=U_2, U_3=U_3,
                total_unigrams=total_unigrams,
                D=args.discount,
                top_k=max(k_values)
            )
            
            # Extract ordered word list
            ranked_words = [cand_tuple[0] for cand_tuple, _ in topk_with_probs]
            
            for k in k_values:
                if answer in ranked_words[:k]:
                    recall_counts[k] += 1
        
        print("\n" + "=" * 50)
        print("  Smoothed Fourgram Candidate Recall@k")
        print("=" * 50)
        for k in k_values:
            recall = recall_counts[k] / total if total > 0 else 0.0
            print(f"  Recall@{k:<4d}  {recall:.4f}  ({recall_counts[k]}/{total})")
        print("=" * 50)
        
        # Also report Top-1 accuracy (equivalent to standalone smoothed fourgram)
        top1_acc = recall_counts[1] / total if total > 0 else 0.0
        print(f"\n  Top-1 accuracy (= standalone smoothed fourgram): {top1_acc:.4f}")
        print(f"  Recall ceiling for hybrid model (top_k={args.top_k}): "
              f"{recall_counts[min(args.top_k, max(k_values))] / total:.4f}")
        return

    # ─── Normal Hybrid Mode (Fourgram + Qwen) ───
    # Load Qwen model
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
    
    predictions = []
    print(f"Predicting with alpha={args.alpha}, discount={args.discount}, top_k={args.top_k}...")
    
    for i in tqdm(range(0, len(contexts), args.batch_size), desc="Hybrid Predicting"):
        batch_contexts = contexts[i:i + args.batch_size]
        batch_letters = first_letters[i:i + args.batch_size]
        
        batch_topk_candidates = []
        for ctx, fl in zip(batch_contexts, batch_letters):
            fl_lower = fl.lower()
            all_cands_for_letter = candidates_by_letter.get(fl_lower, []) # type: ignore
            
            if not all_cands_for_letter:
                batch_topk_candidates.append([])
                continue
                
            # Use Smoothed Fourgram (Absolute Discounting) to rank and return top_k with probabilities
            topk_with_probs = smoothed_fourgram_topk(
                context=ctx,
                candidates=all_cands_for_letter,
                unigrams=unigrams,
                bigrams=bigrams,
                trigrams=trigrams,
                fourgrams=fourgrams,
                U_1=U_1, U_2=U_2, U_3=U_3,
                total_unigrams=total_unigrams,
                D=args.discount,
                top_k=args.top_k
            )
            batch_topk_candidates.append(topk_with_probs)
            
        batch_preds = predict_batch_hybrid(model, tokenizer, batch_contexts, batch_letters, 
                                           batch_topk_candidates, args.alpha)
        predictions.extend(batch_preds)
        
    save_predictions(dev_df, predictions, args.output)
    print(f"Predictions saved to {args.output}")
    
    if 'answer' in dev_df.columns:
        acc, correct, total = compute_accuracy(predictions, dev_df)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
