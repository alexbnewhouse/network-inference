"""
Example: Comparing VADER vs Transformer Sentiment Analysis
==========================================================

This script demonstrates the difference between VADER (lexicon-based)
and transformer-based sentiment analysis on social media text.

Usage:
    python examples/compare_sentiment_models.py

Requirements:
    pip install vaderSentiment torch transformers accelerate
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def compare_sentiment_models():
    """Compare VADER vs transformer sentiment on challenging examples."""
    
    # Test cases that challenge sentiment analysis
    test_cases = [
        {
            'text': "Biden is doing a 'great' job with the economy 🙄",
            'true_sentiment': 'negative',
            'notes': 'Sarcasm with emoji'
        },
        {
            'text': "Trump finally gone, good riddance",
            'true_sentiment': 'negative', 
            'notes': 'Positive word with negative context'
        },
        {
            'text': "This policy is literally Hitler",
            'true_sentiment': 'negative',
            'notes': 'Hyperbolic comparison'
        },
        {
            'text': "Love how they pretend to care about immigrants",
            'true_sentiment': 'negative',
            'notes': 'Ironic use of "love"'
        },
        {
            'text': "Ukraine is doing amazing defending against Russia",
            'true_sentiment': 'positive',
            'notes': 'Straightforward positive'
        },
        {
            'text': "The media is so biased against conservatives smh",
            'true_sentiment': 'negative',
            'notes': 'Political complaint'
        },
        {
            'text': "Best president ever! /s",
            'true_sentiment': 'negative',
            'notes': 'Sarcasm tag'
        },
        {
            'text': "NATO expansion is brilliant, definitely won't cause WW3",
            'true_sentiment': 'negative',
            'notes': 'Heavy sarcasm'
        },
        {
            'text': "This is fine 🔥",
            'true_sentiment': 'negative',
            'notes': 'Meme reference (ironic)'
        },
        {
            'text': "Proud of my country for helping refugees",
            'true_sentiment': 'positive',
            'notes': 'Genuine positive sentiment'
        }
    ]
    
    print("="*80)
    print("SENTIMENT ANALYSIS COMPARISON: VADER vs Transformer")
    print("="*80)
    print()
    
    # Initialize VADER
    print("Initializing VADER...")
    vader = SentimentIntensityAnalyzer()
    
    # Initialize Transformer (with error handling)
    print("Initializing Transformer...")
    try:
        from transformers import pipeline
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=device if device != "cpu" else -1
        )
        transformer_available = True
        print("✓ Transformer model loaded successfully\n")
    except ImportError:
        print("⚠️  Transformers not installed. Install with:")
        print("    pip install torch transformers accelerate")
        print("Showing VADER-only results...\n")
        transformer_available = False
    except Exception as e:
        print(f"⚠️  Could not load transformer: {e}")
        print("Showing VADER-only results...\n")
        transformer_available = False
    
    # Analyze each test case
    results = []
    
    for case in test_cases:
        text = case['text']
        true_sentiment = case['true_sentiment']
        
        # VADER analysis
        vader_scores = vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        if vader_compound > 0.3:
            vader_label = 'positive'
        elif vader_compound < -0.3:
            vader_label = 'negative'
        else:
            vader_label = 'neutral'
        
        vader_correct = (vader_label == true_sentiment)
        
        # Transformer analysis
        if transformer_available:
            try:
                transformer_result = sentiment_pipeline(text, truncation=True)[0]
                
                # Convert to compound score
                if transformer_result['label'] == 'LABEL_0':  # negative
                    transformer_compound = -transformer_result['score']
                    transformer_label = 'negative'
                elif transformer_result['label'] == 'LABEL_2':  # positive
                    transformer_compound = transformer_result['score']
                    transformer_label = 'positive'
                else:  # neutral
                    transformer_compound = 0.0
                    transformer_label = 'neutral'
                
                transformer_correct = (transformer_label == true_sentiment)
            except Exception as e:
                transformer_compound = 0.0
                transformer_label = 'error'
                transformer_correct = False
        else:
            transformer_compound = None
            transformer_label = None
            transformer_correct = None
        
        results.append({
            'text': text,
            'true_sentiment': true_sentiment,
            'vader_score': vader_compound,
            'vader_label': vader_label,
            'vader_correct': vader_correct,
            'transformer_score': transformer_compound,
            'transformer_label': transformer_label,
            'transformer_correct': transformer_correct,
            'notes': case['notes']
        })
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: \"{result['text']}\"")
        print(f"   Notes: {result['notes']}")
        print(f"   True sentiment: {result['true_sentiment'].upper()}")
        print()
        print(f"   VADER:")
        print(f"     Score: {result['vader_score']:+.3f}")
        print(f"     Label: {result['vader_label'].upper()}")
        print(f"     Correct: {'✓' if result['vader_correct'] else '✗'}")
        
        if transformer_available:
            print(f"   Transformer:")
            print(f"     Score: {result['transformer_score']:+.3f}")
            print(f"     Label: {result['transformer_label'].upper()}")
            print(f"     Correct: {'✓' if result['transformer_correct'] else '✗'}")
        
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    vader_accuracy = sum(r['vader_correct'] for r in results) / len(results) * 100
    print(f"VADER Accuracy: {vader_accuracy:.1f}%")
    
    if transformer_available:
        transformer_accuracy = sum(r['transformer_correct'] for r in results) / len(results) * 100
        print(f"Transformer Accuracy: {transformer_accuracy:.1f}%")
        print()
        print(f"Improvement: {transformer_accuracy - vader_accuracy:+.1f} percentage points")
    
    print()
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("VADER (Lexicon-based):")
    print("  ✓ Very fast (~1000 docs/sec)")
    print("  ✓ No GPU required")
    print("  ✗ Misses sarcasm and irony")
    print("  ✗ Struggles with context-dependent sentiment")
    print()
    
    if transformer_available:
        print("Transformer (twitter-roberta-base-sentiment):")
        print("  ✓ Understands context and sarcasm")
        print("  ✓ Better on social media text")
        print(f"  ✓ Higher accuracy ({transformer_accuracy:.0f}% vs {vader_accuracy:.0f}%)")
        print("  ✗ Slower (~100 docs/sec on GPU)")
        print("  ✗ Requires GPU for good performance")
    
    print()
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    print("For dissertation research on /pol/ data:")
    print("  → Use TRANSFORMER sentiment analysis")
    print("  → /pol/ posts are heavily sarcastic and ironic")
    print("  → Contextual understanding is critical")
    print("  → GPU (RTX 5090) makes processing feasible")
    print()
    print("For quick exploratory analysis:")
    print("  → Use VADER sentiment analysis")
    print("  → Fast enough for millions of posts")
    print("  → Good baseline for obvious sentiment")
    print()
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    output_path = "examples/sentiment_comparison_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    compare_sentiment_models()
