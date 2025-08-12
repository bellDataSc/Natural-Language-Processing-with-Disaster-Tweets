"""
Data loading utilities for the Disaster Tweets NLP project
Author: Isabel Cruz (@bellDataSc)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from config import TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE
except ImportError:
 
    TRAIN_FILE = Path("data/raw/train.csv")
    TEST_FILE = Path("data/raw/test.csv")
    SAMPLE_SUBMISSION_FILE = Path("data/raw/sample_submission.csv")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loading and basic validation utilities for disaster tweets classification
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to data directory. If None, uses default from config.
        """
        self.data_dir = data_dir
        
    def load_train_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load training data from CSV file
        
        Args:
            file_path: Path to training data file. If None, uses default from config.
            
        Returns:
            pd.DataFrame: Training data
        """
        file_path = file_path or TRAIN_FILE
        
        try:
            logger.info(f"Loading training data from {file_path}")
            train_data = pd.read_csv(file_path)
            logger.info(f"Training data loaded successfully. Shape: {train_data.shape}")
            
            # Basic validation
            self._validate_train_data(train_data)
            
            return train_data
            
        except FileNotFoundError:
            logger.error(f"Training file not found: {file_path}")
            raise FileNotFoundError(
                f"Training file not found: {file_path}\n"
                "Please download the data from Kaggle:\n"
                "kaggle competitions download -c nlp-getting-started"
            )
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def load_test_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load test data from CSV file
        
        Args:
            file_path: Path to test data file. If None, uses default from config.
            
        Returns:
            pd.DataFrame: Test data
        """
        file_path = file_path or TEST_FILE
        
        try:
            logger.info(f"Loading test data from {file_path}")
            test_data = pd.read_csv(file_path)
            logger.info(f"Test data loaded successfully. Shape: {test_data.shape}")
            
            # Basic validation
            self._validate_test_data(test_data)
            
            return test_data
            
        except FileNotFoundError:
            logger.error(f"Test file not found: {file_path}")
            raise FileNotFoundError(
                f"Test file not found: {file_path}\n"
                "Please download the data from Kaggle:\n"
                "kaggle competitions download -c nlp-getting-started"
            )
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    
    def load_sample_submission(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load sample submission file
        
        Args:
            file_path: Path to sample submission file. If None, uses default from config.
            
        Returns:
            pd.DataFrame: Sample submission format
        """
        file_path = file_path or SAMPLE_SUBMISSION_FILE
        
        try:
            logger.info(f"Loading sample submission from {file_path}")
            sample_submission = pd.read_csv(file_path)
            logger.info(f"Sample submission loaded successfully. Shape: {sample_submission.shape}")
            
            return sample_submission
            
        except FileNotFoundError:
            logger.error(f"Sample submission file not found: {file_path}")
            raise FileNotFoundError(
                f"Sample submission file not found: {file_path}\n"
                "Please download the data from Kaggle:\n"
                "kaggle competitions download -c nlp-getting-started"
            )
        except Exception as e:
            logger.error(f"Error loading sample submission: {str(e)}")
            raise
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required datasets
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_data, test_data, sample_submission)
        """
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        sample_submission = self.load_sample_submission()
        
        logger.info("All datasets loaded successfully")
        self._print_data_summary(train_data, test_data)
        
        return train_data, test_data, sample_submission
    
    def _validate_train_data(self, data: pd.DataFrame) -> None:
        """Validate training data structure and content"""
        required_columns = ['id', 'text', 'target']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Training data is missing required columns: {missing_columns}")
        
        # Check for empty texts
        empty_texts = data['text'].isna().sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty texts in training data")
        
        # Check target distribution
        target_distribution = data['target'].value_counts()
        logger.info(f"Target distribution: {target_distribution.to_dict()}")
    
    def _validate_test_data(self, data: pd.DataFrame) -> None:
        """Validate test data structure and content"""
        required_columns = ['id', 'text']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Test data is missing required columns: {missing_columns}")
        
        # Check for empty texts
        empty_texts = data['text'].isna().sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty texts in test data")
    
    def _print_data_summary(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Print summary of loaded data"""
        logger.info("\n" + "="*50)
        logger.info("DATA SUMMARY")
        logger.info("="*50)
        logger.info(f"Training samples: {len(train_data):,}")
        logger.info(f"Test samples: {len(test_data):,}")
        logger.info(f"Total samples: {len(train_data) + len(test_data):,}")
        
        # Class distribution
        class_dist = train_data['target'].value_counts(normalize=True)
        logger.info(f"Class 0 (Non-disaster): {class_dist[0]:.1%}")
        logger.info(f"Class 1 (Disaster): {class_dist[1]:.1%}")
        logger.info("="*50)


# Convenience functions for backward compatibility
def load_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data (backward compatibility function)
    
    Args:
        data_dir: Path to data directory (unused, kept for compatibility)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    loader = DataLoader()
    train_data = loader.load_train_data()
    test_data = loader.load_test_data()
    
    return train_data, test_data


if __name__ == "__main__":
    # Test data loading
    try:
        loader = DataLoader()
        print("DataLoader created successfully!")
        print("To test loading, ensure you have Kaggle data in data/raw/")
        
    except Exception as e:
        print(f"DataLoader test failed: {str(e)}")
        print("\nTo fix this issue:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Setup Kaggle credentials: ~/.kaggle/kaggle.json")
        print("3. Download data: kaggle competitions download -c nlp-getting-started")
        print("4. Extract data to data/raw/ directory")
