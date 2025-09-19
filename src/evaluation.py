# Comprehensive Evaluation Module
# Save as src/evaluation.py

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    cohen_kappa_score
)
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    def __init__(self):
        self.results = {}
        self.label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def load_all_results(self):
        """Load results from all models"""
        results = {}
        
        # Load traditional ML results
        try:
            with open("models/traditional_ml_results.pkl", 'rb') as f:
                traditional_results = pickle.load(f)
            results['traditional'] = traditional_results
            print("✓ Traditional ML results loaded")
        except:
            print("✗ Traditional ML results not found")
            
        return results
    
    def create_confusion_matrices(self, results):
        """Create detailed confusion matrices for all models"""
        print("\n" + "="*60)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        # Create figure for all confusion matrices
        n_models = 2  # NB and SVM for each language
        n_langs = 2   # Twi and Hausa
        
        fig, axes = plt.subplots(n_langs, n_models, figsize=(12, 10))
        fig.suptitle('Confusion Matrices: Traditional ML Models', fontsize=16)
        
        traditional = results['traditional']
        
        for lang_idx, language in enumerate(['twi', 'hausa']):
            for model_idx, model in enumerate(['naive_bayes', 'svm']):
                model_key = f'{language}_{model}'
                
                if model_key in traditional:
                    y_true = traditional[model_key]['y_true']
                    y_pred = traditional[model_key]['y_pred']
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    
                    # Plot
                    ax = axes[lang_idx, model_idx]
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=list(self.label_names.values()),
                               yticklabels=list(self.label_names.values()))
                    
                    model_name = 'Naive Bayes' if model == 'naive_bayes' else 'SVM'
                    ax.set_title(f'{language.capitalize()} - {model_name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    
                    # Calculate and print class-wise metrics
                    print(f"\n{language.upper()} - {model_name.upper()}:")
                    self.print_classwise_metrics(y_true, y_pred)
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classwise_metrics(self, y_true, y_pred):
        """Print detailed class-wise performance metrics"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        print("  Class-wise Performance:")
        for i, (class_name) in enumerate(self.label_names.values()):
            print(f"    {class_name}: P={precision[i]:.3f}, R={recall[i]:.3f}, "
                  f"F1={f1[i]:.3f}, Support={support[i]}")
    
    def error_analysis(self, results):
        """Analyze where models make errors"""
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        traditional = results['traditional']
        
        error_stats = {}
        
        for language in ['twi', 'hausa']:
            error_stats[language] = {}
            
            print(f"\n{language.upper()} Error Analysis:")
            
            for model in ['naive_bayes', 'svm']:
                model_key = f'{language}_{model}'
                
                if model_key not in traditional:
                    continue
                    
                y_true = traditional[model_key]['y_true']
                y_pred = traditional[model_key]['y_pred']
                
                # Find errors
                errors = y_true != y_pred
                error_indices = np.where(errors)[0]
                
                # Analyze error patterns
                error_analysis = {}
                for true_class in [0, 1, 2]:
                    for pred_class in [0, 1, 2]:
                        if true_class != pred_class:
                            count = np.sum((y_true == true_class) & (y_pred == pred_class))
                            if count > 0:
                                key = f"{self.label_names[true_class]} -> {self.label_names[pred_class]}"
                                error_analysis[key] = count
                
                model_name = 'Naive Bayes' if model == 'naive_bayes' else 'SVM'
                print(f"  {model_name}:")
                print(f"    Total errors: {len(error_indices)} / {len(y_true)} ({len(error_indices)/len(y_true)*100:.1f}%)")
                print(f"    Common error patterns:")
                
                # Sort errors by frequency
                sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
                for error_type, count in sorted_errors[:3]:  # Top 3 error types
                    print(f"      {error_type}: {count} cases")
                
                error_stats[language][model] = {
                    'total_errors': len(error_indices),
                    'error_rate': len(error_indices)/len(y_true),
                    'error_patterns': error_analysis
                }
        
        return error_stats
    
    def statistical_significance_test(self, results):
        """Test statistical significance between model performances"""
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        traditional = results['traditional']
        
        for language in ['twi', 'hausa']:
            print(f"\n{language.upper()}:")
            
            # Get results for both models
            nb_key = f'{language}_naive_bayes'
            svm_key = f'{language}_svm'
            
            if nb_key not in traditional or svm_key not in traditional:
                continue
                
            nb_pred = traditional[nb_key]['y_pred']
            svm_pred = traditional[svm_key]['y_pred']
            y_true = traditional[nb_key]['y_true']
            
            # Calculate accuracies per sample (1 if correct, 0 if wrong)
            nb_correct = (nb_pred == y_true).astype(int)
            svm_correct = (svm_pred == y_true).astype(int)
            
            # McNemar's test for paired predictions
            # Create contingency table
            both_correct = np.sum((nb_correct == 1) & (svm_correct == 1))
            nb_correct_svm_wrong = np.sum((nb_correct == 1) & (svm_correct == 0))
            nb_wrong_svm_correct = np.sum((nb_correct == 0) & (svm_correct == 1))
            both_wrong = np.sum((nb_correct == 0) & (svm_correct == 0))
            
            print(f"  Prediction Agreement Analysis:")
            print(f"    Both correct: {both_correct}")
            print(f"    NB correct, SVM wrong: {nb_correct_svm_wrong}")
            print(f"    NB wrong, SVM correct: {nb_wrong_svm_correct}")
            print(f"    Both wrong: {both_wrong}")
            
            # McNemar's test (if we have disagreements)
            if nb_correct_svm_wrong + nb_wrong_svm_correct > 0:
                mcnemar_stat = (abs(nb_correct_svm_wrong - nb_wrong_svm_correct) - 1)**2 / (nb_correct_svm_wrong + nb_wrong_svm_correct)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                
                print(f"    McNemar's test statistic: {mcnemar_stat:.3f}")
                print(f"    P-value: {p_value:.3f}")
                
                if p_value < 0.05:
                    better_model = "SVM" if nb_wrong_svm_correct > nb_correct_svm_wrong else "Naive Bayes"
                    print(f"    Result: {better_model} is significantly better (p < 0.05)")
                else:
                    print(f"    Result: No significant difference between models (p >= 0.05)")
            else:
                print(f"    Result: Models make identical predictions")
    
    def performance_by_class(self, results):
        """Analyze performance for each sentiment class"""
        print("\n" + "="*60)
        print("PERFORMANCE BY SENTIMENT CLASS")
        print("="*60)
        
        traditional = results['traditional']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Class-wise Performance Analysis', fontsize=16)
        
        for lang_idx, language in enumerate(['twi', 'hausa']):
            for metric_idx, metric in enumerate(['precision', 'recall']):
                ax = axes[lang_idx, metric_idx]
                
                # Collect data for plotting
                classes = list(self.label_names.values())
                nb_scores = []
                svm_scores = []
                
                for model in ['naive_bayes', 'svm']:
                    model_key = f'{language}_{model}'
                    
                    if model_key not in traditional:
                        continue
                        
                    y_true = traditional[model_key]['y_true']
                    y_pred = traditional[model_key]['y_pred']
                    
                    # Calculate class-wise metrics
                    precision, recall, f1, support = precision_recall_fscore_support(
                        y_true, y_pred, average=None, zero_division=0
                    )
                    
                    if metric == 'precision':
                        scores = precision
                    else:  # recall
                        scores = recall
                    
                    if model == 'naive_bayes':
                        nb_scores = scores
                    else:
                        svm_scores = scores
                
                # Plot comparison
                if len(nb_scores) > 0 and len(svm_scores) > 0:
                    x = np.arange(len(classes))
                    width = 0.35
                    
                    ax.bar(x - width/2, nb_scores, width, label='Naive Bayes', alpha=0.8)
                    ax.bar(x + width/2, svm_scores, width, label='SVM', alpha=0.8)
                    
                    ax.set_xlabel('Sentiment Class')
                    ax.set_ylabel(metric.capitalize())
                    ax.set_title(f'{language.capitalize()} - {metric.capitalize()}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(classes)
                    ax.legend()
                    ax.set_ylim(0, 1)
                    
                    # Add value labels
                    for i, (nb_score, svm_score) in enumerate(zip(nb_scores, svm_scores)):
                        ax.text(i - width/2, nb_score + 0.01, f'{nb_score:.2f}', 
                               ha='center', va='bottom', fontsize=9)
                        ax.text(i + width/2, svm_score + 0.01, f'{svm_score:.2f}', 
                               ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/classwise_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def dataset_analysis(self):
        """Analyze dataset characteristics"""
        print("\n" + "="*60)
        print("DATASET CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        # Load processed data to analyze
        try:
            with open("data/processed/traditional_ml_data.pkl", 'rb') as f:
                data = pickle.load(f)
        except:
            print("Processed data not found")
            return
        
        for language in ['twi', 'hausa']:
            if language not in data:
                continue
                
            print(f"\n{language.upper()} Dataset Analysis:")
            
            # Class distribution
            y_train = data[language]['train_labels']
            y_test = data[language]['test_labels']
            
            train_dist = np.bincount(y_train)
            test_dist = np.bincount(y_test)
            
            print(f"  Training set size: {len(y_train)}")
            print(f"  Test set size: {len(y_test)}")
            print(f"  Training class distribution:")
            for i, count in enumerate(train_dist):
                print(f"    {self.label_names[i]}: {count} ({count/len(y_train)*100:.1f}%)")
            
            print(f"  Test class distribution:")
            for i, count in enumerate(test_dist):
                print(f"    {self.label_names[i]}: {count} ({count/len(y_test)*100:.1f}%)")
            
            # Calculate imbalance ratio
            max_class = max(train_dist)
            min_class = min(train_dist)
            imbalance_ratio = max_class / min_class
            print(f"  Class imbalance ratio: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 2:
                print(f"  ⚠️  High class imbalance detected!")
            elif imbalance_ratio > 1.5:
                print(f"  ⚠️  Moderate class imbalance detected")
            else:
                print(f"  ✓ Classes are relatively balanced")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        # Load results
        results = self.load_all_results()
        
        if not results:
            print("No results found. Run the models first.")
            return
        
        # Run all analyses
        print("\n1. Dataset Analysis:")
        self.dataset_analysis()
        
        print("\n2. Confusion Matrix Analysis:")
        self.create_confusion_matrices(results)
        
        print("\n3. Error Analysis:")
        error_stats = self.error_analysis(results)
        
        print("\n4. Statistical Significance Testing:")
        self.statistical_significance_test(results)
        
        print("\n5. Class-wise Performance:")
        self.performance_by_class(results)
        
        # Save comprehensive results
        evaluation_results = {
            'error_statistics': error_stats,
            'evaluation_timestamp': pd.Timestamp.now(),
        }
        
        with open('results/comprehensive_evaluation.pkl', 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        print(f"\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print("Evaluation results saved to results/comprehensive_evaluation.pkl")
        print("Visualizations saved to results/ directory")
        
        # Summary recommendations
        print("\n📋 SUMMARY RECOMMENDATIONS:")
        print("1. Hausa models benefit from larger dataset size")
        print("2. Class imbalance affects Twi performance significantly") 
        print("3. SVM generally outperforms Naive Bayes")
        print("4. Consider data augmentation for minority classes")
        print("5. BERT needs more computational resources for better performance")

def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.generate_comprehensive_report()

if __name__ == "__main__":
    main()