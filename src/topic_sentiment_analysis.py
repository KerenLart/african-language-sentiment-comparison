import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class TopicSentimentAnalyzer:
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.topic_models = {}
        self.vectorizers = {}
        self.topic_labels = {}
        self.results = {}
        self.sentiment_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def load_data(self):
        """Load original text data and model predictions"""
        # Load BERT processed data 
        try:
            with open("data/processed/bert_data.pkl", 'rb') as f:
                bert_data = pickle.load(f)
            print("✓ Text data loaded from BERT preprocessing")
        except:
            print("✗ Could not load text data")
            return None
            
        # Load model predictions
        try:
            with open("models/traditional_ml_results.pkl", 'rb') as f:
                predictions = pickle.load(f)
            print("✓ Model predictions loaded")
        except:
            print("✗ Could not load model predictions") 
            return None
            
        return bert_data, predictions
    
    def preprocess_for_topics(self, texts):
        """Clean texts specifically for topic modeling"""
        cleaned_texts = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            text = str(text).lower()
            
            # Remove URLs, mentions, hashtags
            text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
            # Remove numbers and punctuation but keep spaces
            text = re.sub(r'[^\w\s]|[\d]', ' ', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Filter out very short texts
            if len(text.split()) >= 3:  # At least 3 words
                cleaned_texts.append(text)
            else:
                cleaned_texts.append("")  # Keep index alignment
        
        return cleaned_texts
    
    def extract_topics(self, texts, language):
        """Extract topics using LDA"""
        print(f"Extracting topics for {language}...")
        
        # Clean texts
        cleaned_texts = self.preprocess_for_topics(texts)
        
        # Remove empty texts for topic modeling
        non_empty_texts = [text for text in cleaned_texts if text.strip()]
        
        if len(non_empty_texts) < 10:
            print(f"Warning: Too few valid texts for topic modeling in {language}")
            return None, None, None
        
        # Create TF-IDF features for topic modeling
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',  # Remove common English stop words
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8  # Word must appear in less than 80% of documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(non_empty_texts)
            print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        except:
            print(f"  Failed to create TF-IDF matrix for {language}")
            return None, None, None
        
        # Fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,  # Reduced for faster computation
            learning_method='batch'
        )
        
        try:
            lda_model.fit(tfidf_matrix)
            print(f"  LDA model fitted successfully")
        except:
            print(f"  Failed to fit LDA model for {language}")
            return None, None, None
        
        # Get topic assignments for all texts (including empty ones)
        topic_assignments = []
        non_empty_idx = 0
        
        for text in cleaned_texts:
            if text.strip():
                # Get topic probabilities for this text
                text_tfidf = vectorizer.transform([text])
                topic_probs = lda_model.transform(text_tfidf)[0]
                dominant_topic = np.argmax(topic_probs)
                topic_assignments.append(dominant_topic)
                non_empty_idx += 1
            else:
                topic_assignments.append(-1)  # Mark empty texts
        
        # Store models
        self.topic_models[language] = lda_model
        self.vectorizers[language] = vectorizer
        
        return lda_model, vectorizer, topic_assignments
    
    def get_topic_keywords(self, language, n_words=8):
        """Extract keywords for each topic"""
        if language not in self.topic_models:
            return {}
            
        lda_model = self.topic_models[language]
        vectorizer = self.vectorizers[language]
        feature_names = vectorizer.get_feature_names_out()
        
        topic_keywords = {}
        
        for topic_idx in range(self.n_topics):
            # Get top words for this topic
            topic_words = lda_model.components_[topic_idx]
            top_word_indices = topic_words.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_keywords[topic_idx] = top_words
        
        return topic_keywords
    
    def create_topic_labels(self, language):
        """Create interpretable labels for topics based on keywords"""
        topic_keywords = self.get_topic_keywords(language)
        
        labels = {}
        for topic_idx, keywords in topic_keywords.items():
            # Create label from top 3 keywords
            label = " + ".join(keywords[:3])
            labels[topic_idx] = f"Topic {topic_idx}: {label}"
            
        self.topic_labels[language] = labels
        return labels
    
    def analyze_topic_sentiment_distribution(self, bert_data, predictions, language):
        """Analyze how sentiment is distributed across topics"""
        print(f"\nAnalyzing topic-sentiment patterns for {language}...")
        
        # Get test data (what our models were evaluated on)
        test_texts = bert_data[language]['test']['texts']
        true_labels = bert_data[language]['test']['labels']
        
        # Extract topics for test texts
        lda_model, vectorizer, topic_assignments = self.extract_topics(test_texts, language)
        
        if lda_model is None:
            print(f"Skipping {language} due to topic modeling issues")
            return
        
        # Create topic labels
        topic_labels = self.create_topic_labels(language)
        
        # Get model predictions for comparison
        model_predictions = {
            'Naive Bayes': predictions[f'{language}_naive_bayes']['y_pred'],
            'SVM': predictions[f'{language}_svm']['y_pred']
        }
        
        # Analyze topic-sentiment relationships
        topic_sentiment_analysis = {}
        
        # Create DataFrame for easier analysis
        analysis_data = []
        
        for i, (text, true_sentiment, topic) in enumerate(zip(test_texts, true_labels, topic_assignments)):
            if topic == -1:  # Skip empty texts
                continue
                
            row = {
                'text': text,
                'true_sentiment': true_sentiment,
                'topic': topic,
                'topic_label': topic_labels.get(topic, f"Topic {topic}")
            }
            
            # Add model predictions
            for model_name, preds in model_predictions.items():
                if i < len(preds):
                    row[f'{model_name}_pred'] = preds[i]
            
            analysis_data.append(row)
        
        df = pd.DataFrame(analysis_data)
        
        if len(df) == 0:
            print(f"No valid data for analysis in {language}")
            return
        
        # Calculate topic-sentiment statistics
        topic_sentiment_stats = {}
        
        for topic in range(self.n_topics):
            topic_data = df[df['topic'] == topic]
            
            if len(topic_data) == 0:
                continue
                
            topic_stats = {
                'total_samples': len(topic_data),
                'sentiment_distribution': topic_data['true_sentiment'].value_counts().to_dict(),
                'model_accuracy': {}
            }
            
            # Calculate model accuracy on this topic
            for model in ['Naive Bayes', 'SVM']:
                if f'{model}_pred' in topic_data.columns:
                    correct = (topic_data['true_sentiment'] == topic_data[f'{model}_pred']).sum()
                    accuracy = correct / len(topic_data) if len(topic_data) > 0 else 0
                    topic_stats['model_accuracy'][model] = accuracy
            
            topic_sentiment_stats[topic] = topic_stats
        
        # Store results
        self.results[language] = {
            'topic_labels': topic_labels,
            'topic_keywords': self.get_topic_keywords(language),
            'topic_sentiment_stats': topic_sentiment_stats,
            'analysis_dataframe': df
        }
        
        return df
    
    def visualize_topic_sentiment_analysis(self, language):
        """Create visualizations for topic-sentiment analysis"""
        if language not in self.results:
            print(f"No results available for {language}")
            return
        
        results = self.results[language]
        df = results['analysis_dataframe']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{language.capitalize()} - Topic-Sentiment Analysis', fontsize=16)
        
        # 1. Topic distribution
        ax1 = axes[0, 0]
        topic_counts = df['topic'].value_counts().sort_index()
        bars = ax1.bar(range(len(topic_counts)), topic_counts.values, alpha=0.8)
        ax1.set_xlabel('Topic')
        ax1.set_ylabel('Number of Texts')
        ax1.set_title('Distribution of Texts Across Topics')
        ax1.set_xticks(range(len(topic_counts)))
        ax1.set_xticklabels([f'T{i}' for i in topic_counts.index])
        
        # Add value labels
        for bar, count in zip(bars, topic_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2., count + 1,
                    str(count), ha='center', va='bottom')
        
        # 2. Sentiment distribution by topic
        ax2 = axes[0, 1]
        
        # Create cross-tabulation
        topic_sentiment_crosstab = pd.crosstab(df['topic'], df['true_sentiment'], normalize='index')
        
        # Plot stacked bar chart
        topic_sentiment_crosstab.plot(kind='bar', stacked=True, ax=ax2, 
                                     color=['lightcoral', 'lightblue', 'lightgreen'])
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Proportion of Sentiments')
        ax2.set_title('Sentiment Distribution by Topic')
        ax2.legend(['Negative', 'Neutral', 'Positive'])
        ax2.set_xticklabels([f'T{i}' for i in range(self.n_topics)], rotation=0)
        
        # 3. Model accuracy by topic
        ax3 = axes[1, 0]
        
        topics = []
        nb_accuracies = []
        svm_accuracies = []
        
        for topic, stats in results['topic_sentiment_stats'].items():
            if stats['total_samples'] >= 10:  # Only show topics with enough samples
                topics.append(f'T{topic}')
                nb_accuracies.append(stats['model_accuracy'].get('Naive Bayes', 0))
                svm_accuracies.append(stats['model_accuracy'].get('SVM', 0))
        
        if topics:
            x = np.arange(len(topics))
            width = 0.35
            
            ax3.bar(x - width/2, nb_accuracies, width, label='Naive Bayes', alpha=0.8)
            ax3.bar(x + width/2, svm_accuracies, width, label='SVM', alpha=0.8)
            
            ax3.set_xlabel('Topic')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Model Accuracy by Topic')
            ax3.set_xticks(x)
            ax3.set_xticklabels(topics)
            ax3.legend()
            ax3.set_ylim(0, 1)
        
        # 4. Topic keywords word cloud
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('Topic Keywords Overview')
        
        # Create text summary of topics
        topic_text = ""
        for topic, keywords in results['topic_keywords'].items():
            topic_text += f"Topic {topic}: {', '.join(keywords[:5])}\\n"
        
        ax4.text(0.1, 0.9, topic_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(f'results/{language}_topic_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self, language):
        """Generate actionable insights from topic-sentiment analysis"""
        if language not in self.results:
            return
            
        print(f"\n" + "="*60)
        print(f"{language.upper()} - TOPIC-SENTIMENT INSIGHTS")
        print("="*60)
        
        results = self.results[language]
        df = results['analysis_dataframe']
        
        # 1. Most polarizing topics
        print("\n1. TOPIC POLARIZATION ANALYSIS:")
        
        for topic, stats in results['topic_sentiment_stats'].items():
            if stats['total_samples'] < 10:
                continue
                
            sentiment_dist = stats['sentiment_distribution']
            total = stats['total_samples']
            
            # Calculate polarization (how much it deviates from uniform distribution)
            neg_pct = sentiment_dist.get(0, 0) / total * 100
            neu_pct = sentiment_dist.get(1, 0) / total * 100
            pos_pct = sentiment_dist.get(2, 0) / total * 100
            
            # Find dominant sentiment
            dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])
            dominant_name = self.sentiment_names[dominant_sentiment[0]]
            dominant_pct = dominant_sentiment[1] / total * 100
            
            print(f"\n   Topic {topic}: {results['topic_labels'][topic]}")
            print(f"   Sample size: {total}")
            print(f"   Sentiment breakdown: Neg={neg_pct:.1f}%, Neu={neu_pct:.1f}%, Pos={pos_pct:.1f}%")
            print(f"   Dominant sentiment: {dominant_name} ({dominant_pct:.1f}%)")
            
            if dominant_pct > 60:
                print(f"   ⚠️  Highly polarized topic - {dominant_name.lower()} sentiment dominates")
        
        # 2. Model performance insights by topic
        print(f"\n2. MODEL PERFORMANCE BY TOPIC:")
        
        for topic, stats in results['topic_sentiment_stats'].items():
            if stats['total_samples'] < 10:
                continue
                
            nb_acc = stats['model_accuracy'].get('Naive Bayes', 0)
            svm_acc = stats['model_accuracy'].get('SVM', 0)
            
            print(f"\n   Topic {topic}: {results['topic_labels'][topic]}")
            print(f"   Naive Bayes accuracy: {nb_acc:.3f}")
            print(f"   SVM accuracy: {svm_acc:.3f}")
            
            if abs(nb_acc - svm_acc) > 0.1:
                better_model = "SVM" if svm_acc > nb_acc else "Naive Bayes"
                print(f"   💡 {better_model} performs significantly better on this topic")
        
        # 3. Challenging topics for models
        print(f"\n3. MOST CHALLENGING TOPICS:")
        
        challenging_topics = []
        for topic, stats in results['topic_sentiment_stats'].items():
            if stats['total_samples'] >= 10:
                avg_accuracy = np.mean(list(stats['model_accuracy'].values()))
                challenging_topics.append((topic, avg_accuracy, stats['total_samples']))
        
        # Sort by accuracy (ascending)
        challenging_topics.sort(key=lambda x: x[1])
        
        print(f"\n   Topics ranked by difficulty (hardest first):")
        for topic, avg_acc, sample_size in challenging_topics:
            print(f"   Topic {topic}: {avg_acc:.3f} avg accuracy ({sample_size} samples)")
            keywords = results['topic_keywords'][topic][:3]
            print(f"   Keywords: {', '.join(keywords)}")
            print()
    
    def compare_languages(self):
        """Compare topic-sentiment patterns between languages"""
        print("\n" + "="*80)
        print("CROSS-LANGUAGE COMPARISON")
        print("="*80)
        
        if 'twi' not in self.results or 'hausa' not in self.results:
            print("Both languages needed for comparison")
            return
        
        print("\n1. DATASET SIZE IMPACT:")
        twi_total = sum(stats['total_samples'] for stats in self.results['twi']['topic_sentiment_stats'].values())
        hausa_total = sum(stats['total_samples'] for stats in self.results['hausa']['topic_sentiment_stats'].values())
        
        print(f"   Twi analyzed samples: {twi_total}")
        print(f"   Hausa analyzed samples: {hausa_total}")
        print(f"   Size ratio (Hausa/Twi): {hausa_total/twi_total:.1f}x")
        
        print("\n2. MODEL PERFORMANCE COMPARISON:")
        
        for model in ['Naive Bayes', 'SVM']:
            twi_accuracies = []
            hausa_accuracies = []
            
            for lang in ['twi', 'hausa']:
                lang_accuracies = []
                for stats in self.results[lang]['topic_sentiment_stats'].values():
                    if stats['total_samples'] >= 10:
                        lang_accuracies.append(stats['model_accuracy'].get(model, 0))
                
                if lang == 'twi':
                    twi_accuracies = lang_accuracies
                else:
                    hausa_accuracies = lang_accuracies
            
            if twi_accuracies and hausa_accuracies:
                twi_avg = np.mean(twi_accuracies)
                hausa_avg = np.mean(hausa_accuracies)
                
                print(f"\n   {model}:")
                print(f"   Twi average topic accuracy: {twi_avg:.3f}")
                print(f"   Hausa average topic accuracy: {hausa_avg:.3f}")
                print(f"   Performance gap: {hausa_avg - twi_avg:.3f}")
    
    def run_complete_analysis(self):
        """Run the complete topic-sentiment analysis"""
        print("Starting Topic-Sentiment Analysis")
        print("="*50)
        
        # Load data
        data = self.load_data()
        if data is None:
            return
            
        bert_data, predictions = data
        
        # Analyze each language
        for language in ['twi', 'hausa']:
            if language in bert_data:
                self.analyze_topic_sentiment_distribution(bert_data, predictions, language)
                self.visualize_topic_sentiment_analysis(language)
                self.generate_insights(language)
        
        # Compare languages
        if len(self.results) == 2:
            self.compare_languages()
        
        # Save results
        with open('results/topic_sentiment_analysis.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"\n" + "="*80)
        print("TOPIC-SENTIMENT ANALYSIS COMPLETE")
        print("="*80)
        print("Results saved to results/topic_sentiment_analysis.pkl")
        print("Visualizations saved to results/ directory")
        
        print("\n KEY TAKEAWAYS:")
        print("• Topic analysis reveals content-specific model performance")
        print("• Certain topics are inherently more challenging for sentiment analysis")
        print("• Dataset size significantly impacts topic modeling quality")
        print("• Different models excel at different types of content")

def main():
    analyzer = TopicSentimentAnalyzer(n_topics=5)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()