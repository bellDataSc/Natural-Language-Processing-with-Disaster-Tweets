
import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
IMG_DIR = PROJECT_ROOT / "img"


for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, IMG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

# Model parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'tfidf_max_features': 10000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_stop_words': 'english',
    'logistic_regression_C': 1.0,
    'logistic_regression_max_iter': 1000
}

# Visualization settings
PLOT_CONFIG = {
    'figsize': (10, 6),
    'style': 'seaborn-v0_8',
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
    'dpi': 300
}

# Text preprocessing settings
PREPROCESSING_CONFIG = {
    'remove_urls': True,
    'remove_mentions': True,
    'remove_hashtags': False,  
    'remove_punctuation': True,
    'convert_lowercase': True,
    'remove_numbers': False,  
    'min_text_length': 3
}
