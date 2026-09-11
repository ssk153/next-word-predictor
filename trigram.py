import argparse
from collections import Counter
import os
from tqdm import tqdm
from utils import (
    load_training_data_lines,
    build_vocabulary,
    read_dev_data,
    save_predictions,
    compute_accuracy,
    filter_candidates,
    save_model,
    load_model
)

def train_trigram_model(train_path: str):
    """Counts unigrams, bigrams, and trigrams efficiently."""
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    
    print("Training Trigram Model (Optimized)...")
    lines = load_training_data_lines(train_path)
    for line in tqdm(lines, desc="Counting n-grams"):
        tokens = line.strip().split()
        if not tokens: continue
        
        # Fast C-level counting without explicit inner python loops
        unigram_counts.update(tokens)
        if len(tokens) >= 2:
            bigram_counts.update(zip(tokens, tokens[1:]))
        if len(tokens) >= 3:
            trigram_counts.update(zip(tokens, tokens[1:], tokens[2:]))
                
    total_unigrams = sum(unigram_counts.values())
    return unigram_counts, bigram_counts, trigram_counts, total_unigrams

def predict_trigram(context: str, candidates: list, 
                    unigram_counts: Counter, bigram_counts: Counter, 
                    trigram_counts: Counter, total_unigrams: int) -> str:
    if not candidates:
        return ""
        
    tokens = str(context).strip().split()
    w_prev = tokens[-1] if len(tokens) >= 1 else None
    w_prev2 = tokens[-2] if len(tokens) >= 2 else None
    
    # Pre-compute context counts OUTSIDE the candidate loop
    c_trigram = bigram_counts.get((w_prev2, w_prev), 0) if w_prev2 and w_prev else 0
    c_bigram = unigram_counts.get(w_prev, 0) if w_prev else 0
    
    best_candidate = candidates[0]
    best_prob = -1.0
    
    # Use standard dict .get() instead of Counter[...]
    for cand in candidates:
        prob = 0.0
        
        # 1. Trigram probability
        if c_trigram > 0:
            prob = trigram_counts.get((w_prev2, w_prev, cand), 0) / c_trigram
        
        # 2. Backoff to bigram
        if prob == 0.0 and c_bigram > 0:
            prob = bigram_counts.get((w_prev, cand), 0) / c_bigram
                
        # 3. Backoff to unigram
        if prob == 0.0:
            prob = unigram_counts.get(cand, 0) / total_unigrams if total_unigrams > 0 else 0.0
            
        if prob > best_prob:
            best_prob = prob
            best_candidate = cand
            
    return best_candidate

def main():
    parser = argparse.ArgumentParser(description="Trigram LM Baseline for Next Word Prediction")
    parser.add_argument("--train", required=True, help="Path to training tokenized file")
    parser.add_argument("--dev", required=True, help="Path to dev/test CSV file")
    parser.add_argument("--output", default="predictions_trigram.csv", help="Output predictions CSV file")
    parser.add_argument("--model_path", default="trigram_model.pkl", help="Path to load/save the trained model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev rows for quick testing")
    args = parser.parse_args()
    
    loaded = load_model(args.model_path)
    if loaded:
        print(f"Loaded trained model from {args.model_path}")
        vocab, unigrams, bigrams, trigrams, total_unigrams = loaded
    else:
        print("Building vocabulary...")
        vocab = build_vocabulary(args.train)
        print(f"Vocabulary size: {len(vocab)}")
        
        unigrams, bigrams, trigrams, total_unigrams = train_trigram_model(args.train)
        print(f"Saving model to {args.model_path}...")
        save_model((vocab, unigrams, bigrams, trigrams, total_unigrams), args.model_path)
    
    print(f"Reading dev data from {args.dev}...")
    dev_df = read_dev_data(args.dev, limit=args.limit)
    
    print("Caching candidate lists by first letter...")
    candidates_by_letter = {}
    
    predictions = []
    print("Predicting...")
    
    # Fast iteration over arrays instead of iterrows
    contexts = dev_df['context'].values
    first_letters = dev_df['first letter'].values
    
    for context, first_letter in tqdm(zip(contexts, first_letters), total=len(dev_df), desc="Predicting"):
        letter_str = str(first_letter)
        if letter_str not in candidates_by_letter:
            candidates_by_letter[letter_str] = filter_candidates(vocab, letter_str)
            
        candidates = candidates_by_letter[letter_str]
        
        pred = predict_trigram(context, candidates, 
                               unigrams, bigrams, trigrams, total_unigrams)
        predictions.append(pred)
        
    save_predictions(dev_df, predictions, args.output)
    print(f"Predictions saved to {args.output}")
    
    if 'answer' in dev_df.columns:
        acc, correct, total = compute_accuracy(predictions, dev_df)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()
