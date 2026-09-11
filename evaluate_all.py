from utils import compare_models_performance

prediction_files = {
    'Trigram': 'predictions_trigram.csv',
    '4-gram (Stupid Backoff)': 'predictions_fourgram.csv',
    '5-gram (Stupid Backoff)': 'predictions_fivegram.csv',
    '4-gram (Interpolated)': 'predictions_interpolated.csv',
    '4-gram (Smoothed)': 'predictions_smoothed.csv'
}

print("Evaluating models...\n")
compare_models_performance('dev_set_final.csv', prediction_files)
