# Comparative Analysis of Machine Learning Methods for African Language Sentiment Analysis

**Author:** [Your Name]  
**Course:** Computational Models for Social Media Mining  
**Institution:** [Your University]  
**Code Available:** https://github.com/KerenLart/african-language-sentiment-comparison.git

## Overview

This project presents a systematic comparison of machine learning approaches for sentiment analysis in African languages, specifically examining Twi and Hausa using the AfriSenti dataset. The study evaluates traditional machine learning methods (Naive Bayes, Support Vector Machines), transformer-based models (BERT), and hybrid ensemble approaches across 4,818 Twi and 22,152 Hausa social media texts.

## Key Findings

- **Baseline Traditional ML Performance:**
  - Twi: SVM 54.7% F1, Naive Bayes 47.1% F1
  - Hausa: SVM 74.3% F1, Naive Bayes 71.9% F1
- **Enhanced Traditional ML Performance:**
  - Twi: Enhanced SVM 57.2% F1, Enhanced Naive Bayes 54.3% F1
  - Hausa: Enhanced SVM 75.1% F1, Enhanced Naive Bayes 72.2% F1
- **BERT Performance:**
  - Twi: 44.0% F1-score
  - Hausa: 68.5% F1-score
- **Enhanced Ensemble Methods:**
  - Twi: Weighted voting 54.9% F1
  - Hausa: Weighted voting 75.4% F1
- **Enhanced preprocessing achieves significant improvements** - up to 5% F1-score gains for traditional ML methods
- **Traditional ML maintains competitive performance** while requiring significantly less computational resources
- **Dataset size critically impacts performance** - Hausa models benefit from 4.6x larger, more balanced dataset

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Original AfriSenti files
│   └── processed/           # Cleaned/preprocessed data
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── traditional_ml.py
│   ├── bert_model.py
│   ├── hybrid_model.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_traditional_ml.ipynb
│   ├── 03_bert_training.ipynb
│   └── 04_comparative_analysis.ipynb
├── results/
│   ├── figures/
│   └── metrics/
└── docs/
    └── paper/
```

## Dataset Characteristics

**AfriSenti Dataset (Muhammad et al., 2023):**
- **Twi:** 4,818 tweets 
  - Positive: 47.3%, Negative: 37.7%, Neutral: 15.1%
- **Hausa:** 22,152 tweets
  - Positive: 33.1%, Negative: 32.6%, Neutral: 34.3%
- **Labels:** positive, negative, neutral sentiment

## Experimental Results

### Baseline vs Enhanced Performance Comparison

**Baseline Traditional ML Results:**
| Language | Method | Accuracy | Precision | Recall | F1-Score | Training Time |
|----------|--------|----------|-----------|--------|----------|---------------|
| Twi | Naive Bayes | 66.1% | 44.4% | 51.1% | 47.1% | <0.01s |
| Twi | SVM | 65.6% | 57.0% | 55.0% | 54.7% | 20.33s |
| Hausa | Naive Bayes | 71.7% | 72.5% | 71.7% | 71.9% | <0.01s |
| Hausa | SVM | 74.2% | 75.3% | 74.2% | 74.3% | 137s |

**Enhanced Traditional ML Results:**
| Language | Method | Accuracy | Precision | Recall | F1-Score | Improvement |
|----------|--------|----------|-----------|--------|----------|-------------|
| Twi | Enhanced NB | 61.2% | 54.2% | 54.6% | 54.3% | +7.2% |
| Twi | Enhanced SVM | 65.5% | 58.0% | 57.1% | 57.2% | +2.5% |
| Twi | Enhanced Ensemble | 65.5% | 55.6% | 55.2% | 54.9% | - |
| Hausa | Enhanced NB | 71.9% | 72.6% | 71.9% | 72.2% | +0.3% |
| Hausa | Enhanced SVM | 75.0% | 75.7% | 75.0% | 75.1% | +0.8% |
| Hausa | Enhanced Ensemble | 75.3% | 75.8% | 75.3% | 75.4% | - |

**BERT Performance:**
| Language | Method | Accuracy | Precision | Recall | F1-Score | Training Time |
|----------|--------|----------|-----------|--------|----------|---------------|
| Twi | BERT | 62.0% | 44.0% | 44.0% | 44.0% | 1560s |
| Hausa | BERT | 68.1% | 68.5% | 68.5% | 68.5% | 6414s |

### Key Observations
1. **Enhanced SVM achieves optimal performance** across both languages
2. **Enhanced preprocessing provides significant improvements** - up to 7.2% F1-score gain for Twi Naive Bayes
3. **Hausa models demonstrate 15-20% superior performance** compared to Twi equivalents due to larger, balanced dataset
4. **Traditional ML offers computational efficiency gains exceeding 99%** compared to BERT
5. **Enhanced traditional ML outperforms BERT** on both languages while requiring minimal computational resources
6. **Neutral class detection remains challenging**, particularly for the smaller Twi dataset
7. **Class imbalance significantly impacts model effectiveness**, especially for Naive Bayes on Twi

## Generated Visualizations

The project includes three key visualizations demonstrating experimental findings:

1. **Confusion Matrices (Figure 1):** Traditional ML model classification patterns for both languages
2. **Enhanced Performance Metrics (Figure 2):** Comprehensive performance comparison across accuracy, F1-score, precision, and recall
3. **Final Model Comparison (Figure 3):** F1-score comparison across all approaches and languages

## Methodology

### Traditional ML Approach
- TF-IDF feature extraction with aggressive text preprocessing
- Hyperparameter optimization via grid search with 3-fold cross-validation
- Combined train/development sets for training, evaluation on test sets

### BERT Approach  
- DistilBERT multilingual model fine-tuning
- Context-preserving preprocessing approach
- Training parameters: 3 epochs, batch size 16, learning rate 2e-5

### Ensemble Methods
- Majority voting and weighted voting strategies
- Weights determined by individual model validation F1-scores

## Installation and Usage

### Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
transformers>=4.18.0
torch>=1.11.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Installation
```bash
# Clone repository
git clone https://github.com/KerenLart/african-language-sentiment-comparison.git
cd african-language-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Traditional ML experiments
python src/traditional_ml.py


# BERT fine-tuning (requires GPU for optimal performance)
python src/bert_training.py 


# Ensemble methods
python src/hybrid_model.py

# Generate comprehensive evaluation
python src/evaluation.py --output results/
```

### Reproducibility
- Fixed random seeds (42) across all experiments
- Identical data splits maintained
- Hyperparameters documented in source code
- Model checkpoints saved for BERT experiments

## Scientific Contributions

This study provides several contributions to African language NLP research:

1. **Systematic Methodology Comparison:** First comprehensive evaluation of traditional ML, BERT, and ensemble approaches for Twi and Hausa sentiment analysis using standardized metrics

2. **Computational Efficiency Analysis:** Demonstrates that traditional ML methods achieve competitive performance while requiring minimal computational resources

3. **Dataset Size Impact Documentation:** Quantifies the significant effect of dataset size on model performance across different approaches

4. **Practical Implementation Guidance:** Provides actionable recommendations for sentiment analysis tool development in resource-constrained African language contexts

## Research Implications

This study demonstrates that traditional machine learning methods remain highly competitive for African language sentiment analysis, particularly in resource-constrained environments. The findings challenge assumptions about transformer superiority and provide practical guidance for developing sentiment analysis tools in African contexts where computational resources may be limited.

For Ghana specifically, the research establishes foundational knowledge for developing Twi-language sentiment analysis capabilities that can support local businesses, government agencies, and researchers in understanding social media discourse.

## Limitations

- **Modest absolute performance scores** reflect inherent challenges of low-resource language sentiment analysis
- **Limited to social media data** - generalization to other text domains requires validation  
- **Computational constraints** prevented extensive BERT hyperparameter optimization
- **Automatic topic interpretation** proved challenging, limiting content-specific analysis depth

## Future Work

1. Enhanced preprocessing pipelines incorporating African language-specific features
2. Cross-language transfer learning between related African languages
3. Community-driven dataset expansion initiatives
4. Deployment of practical sentiment analysis tools for Ghanaian applications

## References

- Muhammad, S. H., et al. (2023). AfriSenti: A Twitter sentiment analysis benchmark for African languages. arXiv preprint arXiv:2302.08956.
- Original AfriSenti Dataset: https://github.com/afrisenti-semeval/afrisent-semeval-2023
- Systematic comparison methodology based on established NLP evaluation frameworks

## Citation

If this work is referenced, please cite:
```
Lartey Keren Kuma. (2025). Comparative Analysis of Machine Learning Methods for African Language 
Sentiment Analysis: A Study of Twi and Hausa Using AfriSenti Data. 
Course Project, Computational Models for Social Media Mining.
```
---

**Note:** This research contributes to the growing field of African language natural language processing and demonstrates practical approaches for sentiment analysis tool development in resource-constrained environments typical of African language technology contexts.