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
