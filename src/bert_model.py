import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTPipeline:
    def __init__(self, model_name='distilbert-base-multilingual-cased'):
        """
        Initialize BERT pipeline
        Using DistilBERT for faster training while maintaining good performance
        """
        self.model_name = model_name
        self.tokenizer = None
        self.models = {}
        self.results = {}
        self.training_history = {}
        
    def load_processed_data(self, data_path="data/processed/bert_data.pkl"):
        """Load preprocessed BERT data"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def setup_model_and_tokenizer(self, num_labels=3):
        """Setup tokenizer and model"""
        print(f"Setting up {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Tokenizer loaded. Vocabulary size: {len(self.tokenizer)}")
        return self.tokenizer
    
    def create_datasets(self, data, language, max_length=128):
        """Create PyTorch datasets for training"""
        print(f"Creating datasets for {language}...")
        
        datasets = {}
        
        for split in ['train', 'dev', 'test']:
            if split in data[language]:
                texts = data[language][split]['texts']
                labels = data[language][split]['labels']
                
                dataset = SentimentDataset(
                    texts=texts,
                    labels=labels,
                    tokenizer=self.tokenizer,
                    max_length=max_length
                )
                
                datasets[split] = dataset
                print(f"  {split}: {len(dataset)} samples")
        
        return datasets
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, datasets, language, output_dir=None):
        """Train BERT model for a specific language"""
        print(f"\nTraining BERT for {language.upper()}...")
        
        if output_dir is None:
            output_dir = f"models/bert_{language}"
        
        # Create model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
            label2id={'negative': 0, 'neutral': 1, 'positive': 2}
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Reduced for faster training
            per_device_train_batch_size=16,  # Adjust based on GPU memory
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=42,
            dataloader_pin_memory=False,  # Helps with memory issues
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['dev'],
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Store model and training info
        self.models[language] = {
            'model': model,
            'trainer': trainer,
            'training_time': training_time,
            'output_dir': output_dir
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Model saved to {output_dir}")
        
        return trainer, training_time
    
    def evaluate_model(self, trainer, test_dataset, language):
        """Evaluate trained model"""
        print(f"Evaluating BERT for {language}...")
        
        # Evaluate on test set
        start_time = time.time()
        eval_results = trainer.evaluate(test_dataset)
        prediction_time = time.time() - start_time
        
        # Get predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Store results
        self.results[language] = {
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'precision': eval_results['eval_precision'],
            'recall': eval_results['eval_recall'],
            'prediction_time': prediction_time,
            'y_true': y_true,
            'y_pred': y_pred,
            'training_time': self.models[language]['training_time']
        }
        
        print(f"  Accuracy: {eval_results['eval_accuracy']:.3f}")
        print(f"  F1-score: {eval_results['eval_f1']:.3f}")
        print(f"  Prediction time: {prediction_time:.3f} seconds")
        
        return eval_results
    
    def train_all_languages(self, data):
        """Train BERT models for all languages"""
        if self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        all_results = {}
        
        for language in ['twi', 'hausa']:
            if language not in data:
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing {language.upper()} with BERT")
            print(f"{'='*60}")
            
            # Create datasets
            datasets = self.create_datasets(data, language)
            
            # Train model
            trainer, training_time = self.train_model(datasets, language)
            
            # Evaluate model
            eval_results = self.evaluate_model(trainer, datasets['test'], language)
            
            all_results[language] = self.results[language]
        
        return all_results
    
    def create_results_table(self):
        """Create results table for BERT models"""
        table_data = []
        
        for language in ['twi', 'hausa']:
            if language in self.results:
                result = self.results[language]
                table_data.append({
                    'Language': language.capitalize(),
                    'Model': 'BERT',
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'Precision': f"{result['precision']:.3f}",
                    'Recall': f"{result['recall']:.3f}",
                    'F1-Score': f"{result['f1']:.3f}",
                    'Train Time (s)': f"{result['training_time']:.2f}",
                    'Pred Time (s)': f"{result['prediction_time']:.3f}"
                })
        
        df = pd.DataFrame(table_data)
        return df
    
    def plot_results(self):
        """Plot BERT results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Performance metrics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        metrics = ['accuracy', 'f1']
        metric_names = ['Accuracy', 'F1-Score']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            languages = []
            scores = []
            
            for language in ['twi', 'hausa']:
                if language in self.results:
                    languages.append(language.capitalize())
                    scores.append(self.results[language][metric])
            
            bars = ax.bar(languages, scores, alpha=0.8, color=['skyblue', 'lightcoral'])
            ax.set_ylabel(name)
            ax.set_title(f'BERT {name}')
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.suptitle('BERT Performance Results')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_dir="models"):
        """Save BERT results"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "bert_results.pkl"), 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"BERT results saved to {output_dir}")

# Simplified training function for memory-constrained environments
def train_bert_simple(language='twi', epochs=2, batch_size=8):
    """
    Simplified BERT training function with reduced memory requirements
    Use this if you encounter CUDA out of memory errors
    """
    print(f"Training simplified BERT for {language}...")
    
    # Load data
    with open("data/processed/bert_data.pkl", 'rb') as f:
        data = pickle.load(f)
    
    if language not in data:
        print(f"No data found for {language}")
        return
    
    # Setup
    model_name = 'distilbert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets with smaller max_length
    train_dataset = SentimentDataset(
        data[language]['train']['texts'],
        data[language]['train']['labels'],
        tokenizer,
        max_length=64  # Reduced for memory
    )
    
    test_dataset = SentimentDataset(
        data[language]['test']['texts'],
        data[language]['test']['labels'],
        tokenizer,
        max_length=64
    )
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # Training arguments with reduced memory usage
    training_args = TrainingArguments(
        output_dir=f"models/bert_{language}_simple",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",  # Don't save intermediate checkpoints
        eval_strategy="no",   # Skip evaluation during training
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Save memory
    )
    
    def compute_metrics_simple(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        return {'accuracy': accuracy, 'f1': f1}
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics_simple,
    )
    
    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate(test_dataset)
    
    print(f"\nResults for {language}:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.3f}")
    print(f"  F1-score: {eval_results['eval_f1']:.3f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    return eval_results

# Main execution
if __name__ == "__main__":
    print("BERT Fine-tuning Pipeline")
    print("="*50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be slow on CPU.")
        print("Consider using Google Colab with GPU for faster training.")
        
        # Use simplified training for CPU
        print("\nUsing simplified training...")
        for lang in ['twi', 'hausa']:
            train_bert_simple(lang, epochs=1, batch_size=4)
    else:
        # Full BERT training with GPU
        pipeline = BERTPipeline()
        
        # Load data
        print("Loading preprocessed data...")
        data = pipeline.load_processed_data()
        
        # Train all models
        results = pipeline.train_all_languages(data)
        
        # Display results
        print("\n" + "="*60)
        print("BERT RESULTS SUMMARY")
        print("="*60)
        
        results_df = pipeline.create_results_table()
        print(results_df.to_string(index=False))
        
        # Plot results
        pipeline.plot_results()
        
        # Save results
        pipeline.save_results()
        
        print("\nBERT training complete!")