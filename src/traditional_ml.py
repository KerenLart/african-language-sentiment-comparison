import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

class TraditionalMLPipeline:
    def __init__(self):
        self.vectorizers = {}
        self.models = {}
        self.results = {}
        
    def load_processed_data(self, data_path="data/processed/traditional_ml_data.pkl"):
        """Load preprocessed data for traditional ML"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def create_tfidf_features(self, train_texts, test_texts, language, max_features=5000):
        """Create TF-IDF features for a language"""
        print(f"Creating TF-IDF features for {language}...")
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigrams and bigrams
            stop_words=None,  # Keep all words for African languages
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True
        )
        
        # Fit on training data and transform both train and test
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        
        # Store vectorizer for this language
        self.vectorizers[language] = vectorizer
        
        print(f"  Feature shape: {train_features.shape}")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        return train_features, test_features
    
    def train_naive_bayes(self, X_train, y_train, language):
        """Train Naive Bayes classifier"""
        print(f"Training Naive Bayes for {language}...")
        
        start_time = time.time()
        
        # Train Multinomial Naive Bayes
        nb_model = MultinomialNB(alpha=1.0)
        nb_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Store model
        model_key = f"{language}_naive_bayes"
        self.models[model_key] = nb_model
        
        print(f"  Training time: {training_time:.2f} seconds")
        
        return nb_model, training_time
    
    def train_svm(self, X_train, y_train, language):
        """Train SVM classifier with hyperparameter tuning"""
        print(f"Training SVM for {language}...")
        
        start_time = time.time()
        
        # For faster training, use a subset for hyperparameter tuning if dataset is large
        if len(y_train) > 5000:
            # Use stratified sample for tuning
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            tune_idx, _ = next(splitter.split(X_train, y_train))
            X_tune = X_train[tune_idx]
            y_tune = y_train[tune_idx]
        else:
            X_tune = X_train
            y_tune = y_train
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        svm_model = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm_model, 
            param_grid, 
            cv=3, 
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_tune, y_tune)
        
        # Train final model on full training data with best parameters
        best_svm = SVC(**grid_search.best_params_, random_state=42)
        best_svm.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Store model
        model_key = f"{language}_svm"
        self.models[model_key] = best_svm
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return best_svm, training_time
    
    def evaluate_model(self, model, X_test, y_test, model_name, language):
        """Evaluate model performance"""
        print(f"Evaluating {model_name} for {language}...")
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        # Store results
        result_key = f"{language}_{model_name}"
        self.results[result_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_time': prediction_time,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(f"  Prediction time: {prediction_time:.3f} seconds")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_time': prediction_time
        }
    
    def train_and_evaluate_all(self, data):
        """Train and evaluate all traditional ML models"""
        all_results = {}
        
        for language in ['twi', 'hausa']:
            if language not in data:
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing {language.upper()}")
            print(f"{'='*50}")
            
            # Get data for this language
            train_texts = data[language]['train_texts']
            train_labels = data[language]['train_labels']
            test_texts = data[language]['test_texts']
            test_labels = data[language]['test_labels']
            
            # Create TF-IDF features
            X_train, X_test = self.create_tfidf_features(
                train_texts, test_texts, language
            )
            
            # Train Naive Bayes
            nb_model, nb_train_time = self.train_naive_bayes(X_train, train_labels, language)
            nb_results = self.evaluate_model(nb_model, X_test, test_labels, 'naive_bayes', language)
            nb_results['training_time'] = nb_train_time
            
            # Train SVM
            svm_model, svm_train_time = self.train_svm(X_train, train_labels, language)
            svm_results = self.evaluate_model(svm_model, X_test, test_labels, 'svm', language)
            svm_results['training_time'] = svm_train_time
            
            # Store results for this language
            all_results[language] = {
                'naive_bayes': nb_results,
                'svm': svm_results
            }
        
        return all_results
    
    def create_results_table(self, results):
        """Create a comprehensive results table"""
        table_data = []
        
        for language in ['twi', 'hausa']:
            if language in results:
                for model in ['naive_bayes', 'svm']:
                    if model in results[language]:
                        result = results[language][model]
                        table_data.append({
                            'Language': language.capitalize(),
                            'Model': 'Naive Bayes' if model == 'naive_bayes' else 'SVM',
                            'Accuracy': f"{result['accuracy']:.3f}",
                            'Precision': f"{result['precision']:.3f}",
                            'Recall': f"{result['recall']:.3f}",
                            'F1-Score': f"{result['f1']:.3f}",
                            'Train Time (s)': f"{result['training_time']:.2f}",
                            'Pred Time (s)': f"{result['prediction_time']:.3f}"
                        })
        
        df = pd.DataFrame(table_data)
        return df
    
    def plot_results(self, results):
        """Create visualization of results"""
        # Create performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Traditional ML Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            
            # Data for plotting
            languages = []
            nb_scores = []
            svm_scores = []
            
            for language in ['twi', 'hausa']:
                if language in results:
                    languages.append(language.capitalize())
                    nb_scores.append(results[language]['naive_bayes'][metric])
                    svm_scores.append(results[language]['svm'][metric])
            
            # Create bar plot
            x = np.arange(len(languages))
            width = 0.35
            
            ax.bar(x - width/2, nb_scores, width, label='Naive Bayes', alpha=0.8)
            ax.bar(x + width/2, svm_scores, width, label='SVM', alpha=0.8)
            
            ax.set_xlabel('Language')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.set_xticks(x)
            ax.set_xticklabels(languages)
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, (nb, svm) in enumerate(zip(nb_scores, svm_scores)):
                ax.text(i - width/2, nb + 0.01, f'{nb:.3f}', ha='center', va='bottom')
                ax.text(i + width/2, svm + 0.01, f'{svm:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Training time comparison
        plt.figure(figsize=(10, 6))
        
        languages = []
        nb_times = []
        svm_times = []
        
        for language in ['twi', 'hausa']:
            if language in results:
                languages.append(language.capitalize())
                nb_times.append(results[language]['naive_bayes']['training_time'])
                svm_times.append(results[language]['svm']['training_time'])
        
        x = np.arange(len(languages))
        plt.bar(x - 0.35/2, nb_times, 0.35, label='Naive Bayes', alpha=0.8)
        plt.bar(x + 0.35/2, svm_times, 0.35, label='SVM', alpha=0.8)
        
        plt.xlabel('Language')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(x, languages)
        plt.legend()
        plt.yscale('log')  # Log scale since SVM might be much slower
        
        # Add value labels
        for i, (nb, svm) in enumerate(zip(nb_times, svm_times)):
            plt.text(i - 0.35/2, nb, f'{nb:.1f}s', ha='center', va='bottom')
            plt.text(i + 0.35/2, svm, f'{svm:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, output_dir="models"):
        """Save trained models and vectorizers"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            with open(os.path.join(output_dir, f"{model_name}.pkl"), 'wb') as f:
                pickle.dump(model, f)
        
        # Save vectorizers
        for lang, vectorizer in self.vectorizers.items():
            with open(os.path.join(output_dir, f"{lang}_vectorizer.pkl"), 'wb') as f:
                pickle.dump(vectorizer, f)
        
        # Save results
        with open(os.path.join(output_dir, "traditional_ml_results.pkl"), 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Models and results saved to {output_dir}")


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TraditionalMLPipeline()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = pipeline.load_processed_data()
    
    # Train and evaluate all models
    print("\nStarting traditional ML training and evaluation...")
    results = pipeline.train_and_evaluate_all(data)
    
    # Create and display results table
    print("\n" + "="*80)
    print("TRADITIONAL ML RESULTS SUMMARY")
    print("="*80)
    
    results_df = pipeline.create_results_table(results)
    print(results_df.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating performance visualizations...")
    pipeline.plot_results(results)
    
    # Save models
    print("\nSaving models...")
    pipeline.save_models()
    
    print("\n" + "="*80)
    print("TRADITIONAL ML PIPELINE COMPLETE!")
    print("="*80)
    
    print("\nKey Findings:")
    for language in ['twi', 'hausa']:
        if language in results:
            print(f"\n{language.upper()}:")
            nb_f1 = results[language]['naive_bayes']['f1']
            svm_f1 = results[language]['svm']['f1']
            best_model = 'Naive Bayes' if nb_f1 > svm_f1 else 'SVM'
            print(f"  Best model: {best_model}")
            print(f"  Naive Bayes F1: {nb_f1:.3f}")
            print(f"  SVM F1: {svm_f1:.3f}")
    
    print("\nNext steps:")
    print("1. Implement BERT fine-tuning")
    print("2. Create hybrid ensemble method")
    print("3. Comparative analysis across all approaches")