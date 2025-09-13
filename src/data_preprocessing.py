# Data Preprocessing Pipeline
# Save as src/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class AfriSentiPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.label_mapping = None
        
    def load_data(self, data_path="data/raw"):
        """Load all AfriSenti datasets"""
        data = {}
        
        for lang in ['twi', 'hausa']:
            data[lang] = {}
            for split in ['train', 'dev', 'test']:
                file_path = os.path.join(data_path, f"{lang}_{split}.tsv")
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    data[lang][split] = df
                    print(f"Loaded {lang} {split}: {len(df)} samples")
                except FileNotFoundError:
                    print(f"Warning: {file_path} not found")
        
        return data
    
    def clean_text(self, text, level='light'):
        """
        Clean text with different levels of preprocessing
        level: 'light' for BERT, 'heavy' for traditional ML
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        if level == 'light':
            # Minimal cleaning for BERT (preserve most context)
            text = re.sub(r'http\S+|www\S+', '[URL]', text)  # Replace URLs
            text = re.sub(r'@\w+', '[USER]', text)  # Replace mentions
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
            
        elif level == 'heavy':
            # More aggressive cleaning for traditional ML
            text = text.lower()
            text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
            text = re.sub(r'@\w+', '', text)  # Remove mentions
            text = re.sub(r'#\w+', '', text)  # Remove hashtags
            text = re.sub(r'\d+', '[NUM]', text)  # Replace numbers
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip()
        
        return text
    
    def encode_labels(self, labels):
        """Encode string labels to integers"""
        if self.label_mapping is None:
            # Fit label encoder
            unique_labels = list(set(labels))
            self.label_encoder.fit(unique_labels)
            self.label_mapping = {
                label: idx for idx, label in enumerate(self.label_encoder.classes_)
            }
            print(f"Label mapping: {self.label_mapping}")
        
        return self.label_encoder.transform(labels)
    
    def prepare_traditional_ml_data(self, data):
        """Prepare data for traditional ML (Naive Bayes, SVM)"""
        processed_data = {}
        
        for lang in ['twi', 'hausa']:
            if lang not in data:
                continue
                
            processed_data[lang] = {}
            
            # Combine train and dev for traditional ML training
            train_df = data[lang]['train'].copy()
            dev_df = data[lang]['dev'].copy()
            combined_train = pd.concat([train_df, dev_df], ignore_index=True)
            
            # Process training data
            train_texts = [self.clean_text(text, 'heavy') for text in combined_train['tweet']]
            train_labels = self.encode_labels(combined_train['label'].tolist())
            
            # Process test data
            test_texts = [self.clean_text(text, 'heavy') for text in data[lang]['test']['tweet']]
            test_labels = self.encode_labels(data[lang]['test']['label'].tolist())
            
            processed_data[lang] = {
                'train_texts': train_texts,
                'train_labels': train_labels,
                'test_texts': test_texts,
                'test_labels': test_labels
            }
            
            print(f"{lang.upper()} Traditional ML data:")
            print(f"  Train: {len(train_texts)} samples")
            print(f"  Test: {len(test_texts)} samples")
        
        return processed_data
    
    def prepare_bert_data(self, data):
        """Prepare data for BERT fine-tuning"""
        processed_data = {}
        
        for lang in ['twi', 'hausa']:
            if lang not in data:
                continue
                
            processed_data[lang] = {}
            
            # Keep original train/dev/test splits for BERT
            for split in ['train', 'dev', 'test']:
                if split in data[lang]:
                    df = data[lang][split].copy()
                    
                    # Light cleaning for BERT
                    texts = [self.clean_text(text, 'light') for text in df['tweet']]
                    labels = self.encode_labels(df['label'].tolist())
                    
                    processed_data[lang][split] = {
                        'texts': texts,
                        'labels': labels,
                        'original_labels': df['label'].tolist()
                    }
            
            print(f"{lang.upper()} BERT data:")
            for split in ['train', 'dev', 'test']:
                if split in processed_data[lang]:
                    print(f"  {split}: {len(processed_data[lang][split]['texts'])} samples")
        
        return processed_data
    
    def save_processed_data(self, traditional_data, bert_data, output_dir="data/processed"):
        """Save processed data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save traditional ML data
        with open(os.path.join(output_dir, "traditional_ml_data.pkl"), 'wb') as f:
            pickle.dump(traditional_data, f)
        
        # Save BERT data
        with open(os.path.join(output_dir, "bert_data.pkl"), 'wb') as f:
            pickle.dump(bert_data, f)
        
        # Save label encoder
        with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save label mapping for reference
        with open(os.path.join(output_dir, "label_mapping.txt"), 'w') as f:
            f.write("Label Mapping:\n")
            for label, idx in self.label_mapping.items():
                f.write(f"{label}: {idx}\n")
        
        print(f"Processed data saved to {output_dir}")
    
    def get_data_statistics(self, data):
        """Print data statistics"""
        print("\n=== DATA STATISTICS ===")
        
        for lang in ['twi', 'hausa']:
            if lang not in data:
                continue
                
            print(f"\n{lang.upper()}:")
            total_samples = 0
            
            for split in ['train', 'dev', 'test']:
                if split in data[lang]:
                    count = len(data[lang][split])
                    total_samples += count
                    print(f"  {split}: {count} samples")
            
            print(f"  Total: {total_samples} samples")
            
            # Label distribution
            all_labels = []
            for split in ['train', 'dev', 'test']:
                if split in data[lang]:
                    all_labels.extend(data[lang][split]['label'].tolist())
            
            label_counts = pd.Series(all_labels).value_counts()
            print(f"  Label distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count} ({count/len(all_labels)*100:.1f}%)")

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = AfriSentiPreprocessor()
    
    # Load raw data
    print("Loading raw data...")
    raw_data = preprocessor.load_data()
    
    # Get statistics
    preprocessor.get_data_statistics(raw_data)
    
    # Prepare data for both approaches
    print("\nPreparing traditional ML data...")
    traditional_data = preprocessor.prepare_traditional_ml_data(raw_data)
    
    print("\nPreparing BERT data...")
    bert_data = preprocessor.prepare_bert_data(raw_data)
    
    # Save processed data
    print("\nSaving processed data...")
    preprocessor.save_processed_data(traditional_data, bert_data)
    
    print("\nPreprocessing complete!")
    print("\nNext steps:")
    print("1. Implement traditional ML models (Naive Bayes, SVM)")
    print("2. Implement BERT fine-tuning")
    print("3. Create evaluation framework")