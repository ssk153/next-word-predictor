import argparse
import torch
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
from tqdm import tqdm
from utils import (
    load_training_data_lines,
    read_dev_data,
    save_predictions,
    compute_accuracy,
    save_model,
    load_model
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def build_filtered_vocabulary(train_path: str, min_freq: int = 2) -> set:
    """Builds a vocabulary and filters out rare words (Improvement 4)."""
    vocab_counter = Counter()
    lines = load_training_data_lines(train_path)
    for line in tqdm(lines, desc="Building filtered vocabulary"):
        tokens = line.strip().split()
        vocab_counter.update(tokens)
        
    filtered_vocab = {word for word, count in vocab_counter.items() if count >= min_freq}
    return filtered_vocab

def precompute_candidates(vocab: set, tokenizer) -> dict:
    """
    Precomputes token IDs for all candidates and groups by first letter.
    (Improvement 3: Candidate Metadata Lookup)
    """
    candidates_by_letter = {}
    for word in tqdm(vocab, desc="Precomputing tokenized candidates"):
        letter = word[0].lower()
        # Qwen tokenizer prefix space to simulate word starts
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if not token_ids:
            continue
            
        if letter not in candidates_by_letter:
            candidates_by_letter[letter] = []
            
        candidates_by_letter[letter].append((word, token_ids[0], token_ids))
        
    return candidates_by_letter

def predict_batch_twostage(model, tokenizer, contexts: List[str], first_letters: List[str], candidates_by_letter: dict) -> List[str]:
    """
    Two-stage prediction (Improvement 1 & 2):
    Stage 1: First Subword Approximation (Fast) -> Top 20
    Stage 2: Full Word Probability for Top 20 (Accurate)
    """
    tokenizer.padding_side = 'left'
    inputs = tokenizer(contexts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
    logprobs_cpu = next_token_logprobs.cpu()
    
    # Store candidates that need Stage 2 evaluation
    multi_token_evals = []
    context_top_cands = []
    
    for b in range(len(contexts)):
        first_letter = first_letters[b].lower()
        candidates = candidates_by_letter.get(first_letter, [])
        
        if not candidates:
            context_top_cands.append([])
            continue
            
        # STAGE 1: Rank by first token score
        scored_cands = []
        for word, first_token_id, token_ids in candidates:
            score = logprobs_cpu[b, first_token_id].item()
            scored_cands.append((score, word, token_ids))
            
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        top_20 = scored_cands[:20]
        context_top_cands.append(top_20)
        
        # Identify multi-token candidates in the top 20 that need Stage 2
        for score, word, token_ids in top_20:
            if len(token_ids) > 1:
                multi_token_evals.append((b, word, token_ids, score))
                
    # STAGE 2: Full-word probability for multi-token candidates
    multi_token_final_scores = {}
    
    if multi_token_evals:
        batch_input_ids = []
        for b, word, token_ids, first_score in multi_token_evals:
            ctx_ids = inputs.input_ids[b].tolist()
            # Remove left padding from ctx_ids
            pad_id = tokenizer.pad_token_id
            non_pad_idx = 0
            while non_pad_idx < len(ctx_ids) and ctx_ids[non_pad_idx] == pad_id:
                non_pad_idx += 1
            clean_ctx_ids = ctx_ids[non_pad_idx:]
            
            # Context + candidate tokens (except the last one which is predicted)
            full_ids = clean_ctx_ids + token_ids[:-1]
            batch_input_ids.append(full_ids)
            
        # Left pad the new batch
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
        mini_batch_size = 4  # Drastically reduced to prevent MPS 20GB OOM limit
        
        for i in range(0, len(input_tensor), mini_batch_size):
            batch_in = input_tensor[i:i+mini_batch_size]
            batch_mask = mask_tensor[i:i+mini_batch_size]
            with torch.no_grad():
                eval_outputs = model(input_ids=batch_in, attention_mask=batch_mask, use_cache=False)
                # Move to CPU immediately to save GPU memory and allow concatenation
                eval_logits_list.append(eval_outputs.logits.cpu())
                
        eval_logits = torch.cat(eval_logits_list, dim=0)
            
        # Compute final scores
        for i, (b, word, token_ids, first_score) in enumerate(multi_token_evals):
            seq_len = len(token_ids) - 1
            targets = token_ids[1:]
            
            cand_logits = eval_logits[i, -seq_len:, :]
            cand_logprobs = torch.nn.functional.log_softmax(cand_logits, dim=-1)
            
            score = first_score
            for t in range(seq_len):
                score += cand_logprobs[t, targets[t]].item()
                
            multi_token_final_scores[(b, word)] = score

    # FINAL SELECTION
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
    parser = argparse.ArgumentParser(description="Qwen LM Baseline for Next Word Prediction")
    parser.add_argument("--train", required=True, help="Path to training tokenized file (for vocabulary)")
    parser.add_argument("--dev", required=True, help="Path to dev/test CSV file")
    parser.add_argument("--output", default="predictions_qwen.csv", help="Output predictions CSV file")
    parser.add_argument("--vocab_path", default="qwen_filtered_vocab.pkl", help="Path to load/save the parsed vocabulary")
    parser.add_argument("--device", default="auto", help="Device to run on (e.g., 'cuda', 'mps', 'cpu', 'auto')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev rows for quick testing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for parallel context processing")
    args = parser.parse_args()
    
    print(f"Loading pretrained model tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    loaded_data = load_model(args.vocab_path)
    if loaded_data:
        print(f"Loaded precomputed candidates from {args.vocab_path}")
        candidates_by_letter = loaded_data
    else:
        print("Building filtered vocabulary...")
        vocab = build_filtered_vocabulary(args.train, min_freq=2)
        print(f"Filtered vocabulary size: {len(vocab)}")
        
        candidates_by_letter = precompute_candidates(vocab, tokenizer)
        print(f"Saving precomputed candidates to {args.vocab_path}...")
        save_model(candidates_by_letter, args.vocab_path)
        
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
    
    predictions = []
    
    print("Predicting...")
    
    contexts = [str(c).strip() for c in dev_df['context'].values]
    first_letters = [str(l) for l in dev_df['first letter'].values]
    
    # Process in batches!
    for i in tqdm(range(0, len(contexts), args.batch_size), desc="Predicting (Batched Two-Stage)"):
        batch_contexts = contexts[i:i + args.batch_size]
        batch_letters = first_letters[i:i + args.batch_size]
        
        batch_preds = predict_batch_twostage(model, tokenizer, batch_contexts, batch_letters, candidates_by_letter)
        predictions.extend(batch_preds)
        
    save_predictions(dev_df, predictions, args.output)
    print(f"Predictions saved to {args.output}")
    
    if 'answer' in dev_df.columns:
        acc, correct, total = compute_accuracy(predictions, dev_df)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
