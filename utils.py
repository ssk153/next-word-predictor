import pandas as pd
import pickle
import os
from typing import List, Set, Tuple

def load_training_data_lines(file_path: str):
    """Returns an iterator over the lines of the training data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def build_vocabulary(file_path: str) -> Set[str]:
    """Builds the vocabulary from the training data file."""
    vocab = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.update(line.strip().split())
    return vocab

def read_dev_data(file_path: str, limit: int = None) -> pd.DataFrame: # type: ignore
    """Reads the development or test data. Uses Parquet for caching to speed up subsequent loads."""
    parquet_path = file_path.replace('.csv', '.parquet')
    
    if os.path.exists(parquet_path):
        print(f"Loading cached {parquet_path} for faster access...")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"Reading {file_path} and caching as Parquet...")
        df = pd.read_csv(file_path, dtype=str)
        # Handle NaN in 'first letter' just in case
        df['first letter'] = df['first letter'].fillna('')
        df.to_parquet(parquet_path, index=False)
        
    if limit is not None:
        df = df.head(limit)
        
    return df

def filter_candidates(vocab: Set[str], first_letter: str) -> List[str]:
    """Filters vocabulary to only include words that start with the given letter."""
    return sorted([word for word in vocab if word.startswith(first_letter)])

def compute_accuracy(predictions: List[str], df: pd.DataFrame) -> Tuple[float, int, int]:
    """Computes accuracy if the 'answer' column is present in the dataframe."""
    if 'answer' not in df.columns:
        return 0.0, 0, len(df)
    
    correct = 0
    total = len(df)
    for pred, ans in zip(predictions, df['answer']):
        if str(pred) == str(ans):
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def save_predictions(df: pd.DataFrame, predictions: List[str], output_path: str):
    """Saves predictions to a CSV file."""
    out_df = pd.DataFrame({
        'context': df['context'],
        'first letter': df['first letter'],
        'prediction': predictions
    })
    out_df.to_csv(output_path, index=False)

def save_model(model_data: tuple, model_path: str):
    """Saves trained model parameters using pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(model_path: str) -> tuple:
    """Loads trained model parameters using pickle."""
    if not os.path.exists(model_path):
        return None # type: ignore
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def performance_metrics(results: dict) -> None:
    """Print a performance metrics table for multiple models."""
    rows = []
    for model_name, metrics in results.items():
        if isinstance(metrics, dict):
            acc = metrics.get('accuracy', 0.0)
            prec = metrics.get('precision', 0.0)
            rec = metrics.get('recall', 0.0)
            f1 = metrics.get('f1', 0.0)
        else:
            acc, prec, rec, f1 = metrics

        rows.append({
            'Model': model_name,
            'Accuracy': f'{float(acc):.4f}',
            'Precision (Macro avg)': f'{float(prec):.4f}',
            'Recall (Macro avg)': f'{float(rec):.4f}',
            'F1-Score (Macro avg)': f'{float(f1):.4f}',
        })

    print(pd.DataFrame(rows, columns=[
        'Model', 'Accuracy', 'Precision (Macro avg)', 'Recall (Macro avg)', 'F1-Score (Macro avg)'
    ]).to_string(index=False))

def compare_models_performance(dev_file: str, prediction_files: dict):
    """
    Computes and prints performance metrics for multiple models from their prediction files.
    
    Args:
        dev_file: Path to the development set CSV file (must contain 'answer' column).
        prediction_files: Dictionary mapping model name to its prediction CSV file path.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    except ImportError:
        print("Please install scikit-learn to compute Precision, Recall, and F1-Score.")
        return

    dev_df = read_dev_data(dev_file)
    if 'answer' not in dev_df.columns:
        print("Development set does not contain 'answer' column. Cannot compute accuracy.")
        return
        
    y_true = dev_df['answer'].astype(str).tolist()
    
    results = {}
    for model_name, pred_file in prediction_files.items():
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file {pred_file} for {model_name} not found.")
            continue
            
        pred_df = pd.read_csv(pred_file)
        
        # Check if the output format has 'prediction' column
        if 'prediction' in pred_df.columns:
            y_pred = pred_df['prediction'].astype(str).tolist()
        elif len(pred_df.columns) >= 3 and 'context' in pred_df.columns:
            y_pred = pred_df.iloc[:, 2].astype(str).tolist()
        else:
            # Fallback to the first column if no header matches
            y_pred = pred_df.iloc[:, 0].astype(str).tolist()
            
        # Ensure lengths match
        if len(y_pred) != len(y_true):
            print(f"Warning: Length mismatch for {model_name}. Expected {len(y_true)}, got {len(y_pred)}.")
            # Truncate to the minimum length for safe computation
            min_len = min(len(y_true), len(y_pred))
            y_true_eval = y_true[:min_len]
            y_pred_eval = y_pred[:min_len]
        else:
            y_true_eval = y_true
            y_pred_eval = y_pred
            
        acc = accuracy_score(y_true_eval, y_pred_eval)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_eval, y_pred_eval, average='macro', zero_division=0
        )
        
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
    performance_metrics(results)