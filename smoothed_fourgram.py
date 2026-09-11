import argparse
from collections import Counter
import os
from tqdm import tqdm
from utils import (
    read_dev_data,
    save_predictions,
    compute_accuracy,
    filter_candidates,
    load_model
)

def predict_linear_interpolation(context: str, candidates: list, 
                                 unigrams: Counter, bigrams: Counter, 
                                 trigrams: Counter, fourgrams: Counter, 
                                 total_unigrams: int, 
                                 l4: float, l3: float, l2: float, l1: float) -> str:
    """Option 2: Use Interpolated N-Gram Probabilities with fixed weights"""
    if not candidates:
        return ""
        
    tokens = str(context).strip().split()
    w_prev = tokens[-1] if len(tokens) >= 1 else None
    w_prev2 = tokens[-2] if len(tokens) >= 2 else None
    w_prev3 = tokens[-3] if len(tokens) >= 3 else None
    
    c_3 = trigrams.get((w_prev3, w_prev2, w_prev), 0) if w_prev3 and w_prev2 and w_prev else 0
    c_2 = bigrams.get((w_prev2, w_prev), 0) if w_prev2 and w_prev else 0
    c_1 = unigrams.get(w_prev, 0) if w_prev else 0
    
    best_candidate = candidates[0]
    best_prob = -1.0
    
    for cand in candidates:
        p_1 = unigrams.get(cand, 0) / total_unigrams if total_unigrams > 0 else 0.0
        p_2 = bigrams.get((w_prev, cand), 0) / c_1 if c_1 > 0 else 0.0
        p_3 = trigrams.get((w_prev2, w_prev, cand), 0) / c_2 if c_2 > 0 else 0.0
        p_4 = fourgrams.get((w_prev3, w_prev2, w_prev, cand), 0) / c_3 if c_3 > 0 else 0.0
        
        prob = l4 * p_4 + l3 * p_3 + l2 * p_2 + l1 * p_1
        
        if prob > best_prob:
            best_prob = prob
            best_candidate = cand
            
    return best_candidate


def predict_absolute_discounting(context: str, candidates: list, 
                                 unigrams: Counter, bigrams: Counter, 
                                 trigrams: Counter, fourgrams: Counter, 
                                 U_1: Counter, U_2: Counter, U_3: Counter,
                                 total_unigrams: int, D: float) -> str:
    """Option 3: Add Smoothing (Interpolated Absolute Discounting)"""
    if not candidates:
        return ""
        
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
    
    best_candidate = candidates[0]
    best_prob = -1.0
    
    for cand in candidates:
        # P1 (Unigram)
        p_1 = unigrams.get(cand, 0) / total_unigrams if total_unigrams > 0 else 0.0
        
        # P2 (Bigram)
        if c_1 > 0:
            p_2 = max(bigrams.get((w_prev, cand), 0) - D, 0) / c_1 + lambda_2 * p_1
        else:
            p_2 = p_1
            
        # P3 (Trigram)
        if c_2 > 0:
            p_3 = max(trigrams.get((w_prev2, w_prev, cand), 0) - D, 0) / c_2 + lambda_3 * p_2
        else:
            p_3 = p_2
            
        # P4 (Fourgram)
        if c_3 > 0:
            p_4 = max(fourgrams.get((w_prev3, w_prev2, w_prev, cand), 0) - D, 0) / c_3 + lambda_4 * p_3
        else:
            p_4 = p_3
            
        if p_4 > best_prob:
            best_prob = p_4
            best_candidate = cand
            
    return best_candidate


def main():
    parser = argparse.ArgumentParser(description="Advanced N-Gram Models (Interpolation & Smoothing)")
    parser.add_argument("--dev", required=True, help="Path to dev/test CSV file")
    parser.add_argument("--output", default="predictions_advanced.csv", help="Output predictions CSV file")
    parser.add_argument("--model_path", default="fourgram_model.pkl", help="Path to the trained fourgram model")
    parser.add_argument("--method", choices=["linear", "absolute_discounting"], default="absolute_discounting",
                        help="Choose 'linear' for Option 2 or 'absolute_discounting' for Option 3.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dev rows for quick testing")
    
    # Linear Interpolation Weights (Option 2)
    parser.add_argument("--l4", type=float, default=0.5, help="Weight for 4-gram in linear interpolation")
    parser.add_argument("--l3", type=float, default=0.3, help="Weight for 3-gram in linear interpolation")
    parser.add_argument("--l2", type=float, default=0.15, help="Weight for 2-gram in linear interpolation")
    parser.add_argument("--l1", type=float, default=0.05, help="Weight for 1-gram in linear interpolation")
    
    # Smoothing Discount (Option 3)
    parser.add_argument("--discount", type=float, default=0.75, help="Discount factor D for Absolute Discounting")
    
    args = parser.parse_args()
    
    print(f"Loading trained model from {args.model_path}")
    loaded = load_model(args.model_path)
    if not loaded:
        print(f"Error: Model file {args.model_path} not found. Train fourgram.py first.")
        return
        
    vocab, unigrams, bigrams, trigrams, fourgrams, total_unigrams = loaded
    
    U_1, U_2, U_3 = None, None, None
    if args.method == "absolute_discounting":
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
    
    print(f"Reading dev data from {args.dev}...")
    dev_df = read_dev_data(args.dev, limit=args.limit)
    
    print("Caching candidate lists by first letter...")
    candidates_by_letter = {}
    
    predictions = []
    print(f"Predicting using method: {args.method}...")
    
    contexts = dev_df['context'].values
    first_letters = dev_df['first letter'].values
    
    for context, first_letter in tqdm(zip(contexts, first_letters), total=len(dev_df), desc="Predicting"):
        letter_str = str(first_letter)
        if letter_str not in candidates_by_letter:
            candidates_by_letter[letter_str] = filter_candidates(vocab, letter_str)
            
        candidates = candidates_by_letter[letter_str]
        
        if args.method == "linear":
            pred = predict_linear_interpolation(
                context, candidates, unigrams, bigrams, trigrams, fourgrams, total_unigrams,
                args.l4, args.l3, args.l2, args.l1
            )
        else:
            pred = predict_absolute_discounting(
                context, candidates, unigrams, bigrams, trigrams, fourgrams, 
                U_1, U_2, U_3, total_unigrams, args.discount
            )
            
        predictions.append(pred)
        
    save_predictions(dev_df, predictions, args.output)
    print(f"Predictions saved to {args.output}")
    
    if 'answer' in dev_df.columns:
        acc, correct, total = compute_accuracy(predictions, dev_df)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()

