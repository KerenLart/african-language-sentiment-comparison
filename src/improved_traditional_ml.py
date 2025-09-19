# Improved Traditional ML with Enhanced Features
# Save as src/improved_traditional_ml.py

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

class ImprovedTraditionalMLPipeline:
    def __init__(self):
        self.pipelines = {}
        self.results = {}
        self.class_weights = {}
        
    def load_enhanced_data(self, data_path="data/processed/enhanced_traditional_ml_data.pkl"):
        """Load enhanced preprocessed data"""
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            print("✓ Enhanced traditional ML data loaded")
            
            # Load class weights
            with open("data/processed/class_weights.pkl", 'rb') as f:
                self.class_weights = pickle.load(f)
            print("✓ Class weights loaded")
            
            return data
        except Exception as e:
            print(f"✗ Error loading enhanced data: {e}")
            return None
    
    def create_enhanced_features(self, train_texts, test_texts, language, max_features=8000):
        """Create enhanced TF-IDF features with better parameters"""
        print(f"Creating enhanced TF-IDF features for {language}...")
        
        # Enhanced TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),  # Include trigrams
            stop_words=None,  # Keep all words for African languages
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            smooth_idf=True,
            norm='l2',
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )
        
        # Fit and transform
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        
        print(f"  Enhanced feature shape: {train_features.shape}")
        print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        return vectorizer, train_features, test_features
    
    def train_enhanced_naive_bayes(self, X_train, y_train, language):
        """Train enhanced Naive Bayes with class weights"""
        print(f"Training enhanced Naive Bayes for {language}...")
        
        start_time = time.time()
        
        # Use class weights to handle imbalance
        class_weight = self.class_weights.get(language, None)
        
        # Grid search for better parameters
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'fit_prior': [True, False]
        }
        
        nb_model = MultinomialNB()
        
        # Grid search with class weight handling
        grid_search = GridSearchCV(
            nb_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        # If we have class weights, apply sample weights
        if class_weight:
            sample_weights = [class_weight[label] for label in y_train]
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        training_time = time.time() - start_time
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return best_model, training_time
    
    def train_enhanced_svm(self, X_train, y_train, language):
        """Train enhanced SVM with class weights and better parameters"""
        print(f"Training enhanced SVM for {language}...")
        
        start_time = time.time()
        
        # Get class weights
        class_weight = self.class_weights.get(language, 'balanced')
        
        # Enhanced parameter grid
        if X_train.shape[0] > 5000:
            # For larger datasets, use subset for parameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        else:
            # For smaller datasets, more conservative
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale']
            }
        
        svm_model = SVC(
            random_state=42,
            class_weight=class_weight,
            probability=True  # Enable probability estimates
        )
        
        grid_search = GridSearchCV(
            svm_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        training_time = time.time() - start_time
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  Training time: {training_time:.2f} seconds")
        
        return best_model, training_time
    
    def create_enhanced_ensemble(self, nb_model, svm_model):
        """Create enhanced ensemble model"""
        # Voting classifier with probability-based voting
        ensemble = VotingClassifier(
            estimators=[
                ('nb', nb_model),
                ('svm', svm_model)
            ],
            voting='soft',  # Use probability-based voting
            weights=[1, 1.2]  # Slight preference for SVM based on typical performance
        )
        
        return ensemble
    
    def evaluate_enhanced_model(self, model, X_test, y_test, model_name, language):
        """Enhanced model evaluation"""
        print(f"Evaluating enhanced {model_name} for {language}...")
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        # Store enhanced results
        result_key = f"{language}_{model_name}"
        self.results[result_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_time': prediction_time,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-score: {f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_time': prediction_time
        }
    
    def train_and_evaluate_enhanced(self, data):
        """Train and evaluate all enhanced models"""
        all_results = {}
        
        for language in ['twi', 'hausa']:
            if language not in data:
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing Enhanced {language.upper()}")
            print(f"{'='*60}")
            
            # Get enhanced data
            train_texts = data[language]['train_texts']
            train_labels = data[language]['train_labels']
            test_texts = data[language]['test_texts']
            test_labels = data[language]['test_labels']
            
            print(f"Training samples: {len(train_texts)}")
            print(f"Test samples: {len(test_texts)}")
            
            # Create enhanced features
            vectorizer, X_train, X_test = self.create_enhanced_features(
                train_texts, test_texts, language
            )
            
            # Train enhanced models
            nb_model, nb_train_time = self.train_enhanced_naive_bayes(X_train, train_labels, language)
            nb_results = self.evaluate_enhanced_model(nb_model, X_test, test_labels, 'enhanced_naive_bayes', language)
            nb_results['training_time'] = nb_train_time
            
            svm_model, svm_train_time = self.train_enhanced_svm(X_train, train_labels, language)
            svm_results = self.evaluate_enhanced_model(svm_model, X_test, test_labels, 'enhanced_svm', language)
            svm_results['training_time'] = svm_train_time
            
            # Create and train enhanced ensemble
            ensemble = self.create_enhanced_ensemble(nb_model, svm_model)
            
            start_time = time.time()
            ensemble.fit(X_train, train_labels)
            ensemble_train_time = time.time() - start_time
            
            ensemble_results = self.evaluate_enhanced_model(ensemble, X_test, test_labels, 'enhanced_ensemble', language)
            ensemble_results['training_time'] = ensemble_train_time
            
            # Save models for this language
            self.save_enhanced_models(nb_model, svm_model, ensemble, vectorizer, language)
            
            # Store results
            all_results[language] = {
                'enhanced_naive_bayes': nb_results,
                'enhanced_svm': svm_results,
                'enhanced_ensemble': ensemble_results
            }
        
        return all_results
    
    def save_enhanced_models(self, nb_model, svm_model, ensemble, vectorizer, language):
        """Save enhanced models"""
        os.makedirs("models", exist_ok=True)
        
        # Save individual models
        with open(f"models/{language}_enhanced_naive_bayes.pkl", 'wb') as f:
            pickle.dump(nb_model, f)
        
        with open(f"models/{language}_enhanced_svm.pkl", 'wb') as f:
            pickle.dump(svm_model, f)
        
        with open(f"models/{language}_enhanced_ensemble.pkl", 'wb') as f:
            pickle.dump(ensemble, f)
        
        # Save enhanced vectorizer
        with open(f"models/{language}_enhanced_vectorizer.pkl", 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"Enhanced models for {language} saved to models/")
    
    def create_enhanced_results_table(self, results):
        """Create enhanced results table"""
        table_data = []
        
        for language in ['twi', 'hausa']:
            if language in results:
                for model in ['enhanced_naive_bayes', 'enhanced_svm', 'enhanced_ensemble']:
                    if model in results[language]:
                        result = results[language][model]
                        model_display_name = model.replace('enhanced_', '').replace('_', ' ').title()
                        
                        table_data.append({
                            'Language': language.capitalize(),
                            'Model': model_display_name,
                            'Accuracy': f"{result['accuracy']:.3f}",
                            'Precision': f"{result['precision']:.3f}",
                            'Recall': f"{result['recall']:.3f}",
                            'F1-Score': f"{result['f1']:.3f}",
                            'Train Time (s)': f"{result['training_time']:.2f}",
                            'Pred Time (s)': f"{result['prediction_time']:.3f}"
                        })
        
        df = pd.DataFrame(table_data)
        return df
    
    def plot_enhanced_results(self, results):
        """Plot enhanced results with comparison to baseline"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Traditional ML Performance', fontsize=16)
        
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            
            # Data for plotting
            languages = []
            nb_scores = []
            svm_scores = []
            ensemble_scores = []
            
            for language in ['twi', 'hausa']:
                if language in results:
                    languages.append(language.capitalize())
                    nb_scores.append(results[language]['enhanced_naive_bayes'][metric])
                    svm_scores.append(results[language]['enhanced_svm'][metric])
                    ensemble_scores.append(results[language]['enhanced_ensemble'][metric])
            
            # Create bar plot
            x = np.arange(len(languages))
            width = 0.25
            
            bars1 = ax.bar(x - width, nb_scores, width, label='Enhanced NB', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x, svm_scores, width, label='Enhanced SVM', alpha=0.8, color='lightcoral')
            bars3 = ax.bar(x + width, ensemble_scores, width, label='Enhanced Ensemble', alpha=0.8, color='lightgreen')
            
            ax.set_xlabel('Language')
            ax.set_ylabel(name)
            ax.set_title(f'Enhanced {name}')
            ax.set_xticks(x)
            ax.set_xticklabels(languages)
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/enhanced_traditional_ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_enhanced_results(self):
        """Save enhanced results"""
        os.makedirs("results", exist_ok=True)
        
        with open("results/enhanced_traditional_ml_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        print("Enhanced results saved to results/enhanced_traditional_ml_results.pkl")

def main():
    print("Enhanced Traditional ML Pipeline")
    print("="*50)
    
    # Initialize enhanced pipeline
    pipeline = ImprovedTraditionalMLPipeline()
    
    # Load enhanced data
    print("Loading enhanced preprocessed data...")
    data = pipeline.load_enhanced_data()
    
    if data is None:
        print("Please run enhanced preprocessing first!")
        return
    
    # Train and evaluate enhanced models
    print("\nStarting enhanced traditional ML training...")
    results = pipeline.train_and_evaluate_enhanced(data)
    
    # Create and display results
    print("\n" + "="*80)
    print("ENHANCED TRADITIONAL ML RESULTS")
    print("="*80)
    
    results_df = pipeline.create_enhanced_results_table(results)
    print(results_df.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating enhanced visualizations...")
    pipeline.plot_enhanced_results(results)
    
    # Save results
    pipeline.save_enhanced_results()
    
    print("\n" + "="*80)
    print("ENHANCED TRADITIONAL ML COMPLETE!")
    print("="*80)
    
    # Performance comparison
    print("\nKey Improvements:")
    for language in ['twi', 'hausa']:
        if language in results:
            best_f1 = max([
                results[language]['enhanced_naive_bayes']['f1'],
                results[language]['enhanced_svm']['f1'],
                results[language]['enhanced_ensemble']['f1']
            ])
            print(f"  {language.capitalize()}: Best F1-score = {best_f1:.3f}")
    
    print("\nEnhancements Applied:")
    print("  ✓ Language-specific text normalization")
    print("  ✓ Enhanced TF-IDF with trigrams")
    print("  ✓ Class weight handling for imbalance")
    print("  ✓ Grid search optimization")
    print("  ✓ Probability-based ensemble voting")
    print("  ✓ Data augmentation for minority classes")

if __name__ == "__main__":
    main()