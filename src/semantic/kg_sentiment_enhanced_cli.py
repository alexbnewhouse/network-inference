"""
CLI for enhanced sentiment analysis on knowledge graphs.

Usage examples:
    # Analyze stance toward entity
    python -m src.semantic.kg_sentiment_enhanced_cli --input data.csv --entity "Russia" --stance
    
    # Compare sentiment across groups
    python -m src.semantic.kg_sentiment_enhanced_cli --input data.csv --entity "Russia" --group-by board
    
    # Analyze sentiment trends over time
    python -m src.semantic.kg_sentiment_enhanced_cli --temporal output/kg_temporal --entity "Russia" --trends
    
Author: AI Assistant  
Date: 2025-10-21
"""

import argparse
import pandas as pd
from pathlib import Path
from src.semantic.kg_sentiment_enhanced import (
    StanceDetector,
    EntityFramingAnalyzer,
    TemporalSentimentAnalyzer,
    analyze_entity_stance_distribution,
    compare_group_sentiment
)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced sentiment analysis for knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument('--input', help='Input CSV file for stance/framing analysis')
    parser.add_argument('--temporal', help='Directory with temporal KG sentiment data')
    parser.add_argument('--text-col', default='text', help='Name of text column')
    
    # Entity to analyze
    parser.add_argument('--entity', required=True, help='Entity to analyze')
    
    # Analysis types
    parser.add_argument('--stance', action='store_true', help='Analyze stance distribution')
    parser.add_argument('--framing', action='store_true', help='Analyze entity framing')
    parser.add_argument('--trends', action='store_true', help='Analyze sentiment trends over time (requires --temporal)')
    parser.add_argument('--shifts', action='store_true', help='Detect sentiment shifts (requires --temporal)')
    parser.add_argument('--group-by', help='Group column for sentiment comparison (e.g., "board", "user_id")')
    
    # Comparison options
    parser.add_argument('--compare-entities', nargs='+', help='Additional entities to compare')
    
    # Output options
    parser.add_argument('--export', help='Export results to CSV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.trends or args.shifts:
        if not args.temporal:
            parser.error("--trends and --shifts require --temporal")
    elif not args.input:
        parser.error("--input required for stance/framing/group analysis")
    
    # === TEMPORAL SENTIMENT ANALYSIS ===
    if args.temporal:
        print(f"Loading temporal sentiment data from {args.temporal}...")
        tsa = TemporalSentimentAnalyzer(args.temporal)
        
        if args.trends:
            print("\n" + "=" * 80)
            print(f"SENTIMENT TREND: {args.entity}")
            print("=" * 80)
            
            trend = tsa.get_entity_sentiment_trend(args.entity)
            print("\nSentiment by period:")
            print(trend.to_string(index=False))
            
            # Calculate statistics
            valid_trend = trend[trend['sentiment'].notna()]
            if not valid_trend.empty:
                print(f"\nOverall statistics:")
                print(f"  Mean sentiment: {valid_trend['sentiment'].mean():.3f}")
                print(f"  Median sentiment: {valid_trend['sentiment'].median():.3f}")
                print(f"  Std deviation: {valid_trend['sentiment'].std():.3f}")
                print(f"  Total mentions: {valid_trend['n_mentions'].sum()}")
            
            if args.export:
                trend.to_csv(args.export, index=False)
                print(f"\n✓ Trend data exported to {args.export}")
        
        if args.shifts:
            print("\n" + "=" * 80)
            print(f"SENTIMENT SHIFTS: {args.entity}")
            print("=" * 80)
            
            shifts = tsa.detect_sentiment_shifts(args.entity, threshold=0.3)
            if shifts:
                print(f"\nDetected {len(shifts)} significant sentiment shifts:")
                for shift in shifts:
                    arrow = "↑" if shift['direction'] == 'positive' else "↓"
                    print(f"\n  {arrow} {shift['period']}")
                    print(f"     {shift['old_sentiment']:.3f} → {shift['new_sentiment']:.3f} "
                          f"(change: {shift['change']:+.3f})")
            else:
                print("\nNo significant sentiment shifts detected.")
        
        if args.compare_entities:
            all_entities = [args.entity] + args.compare_entities
            print("\n" + "=" * 80)
            print(f"ENTITY COMPARISON")
            print("=" * 80)
            
            comparison = tsa.compare_entity_sentiment(all_entities)
            print("\n" + comparison.to_string(index=False))
            
            if args.export:
                export_path = args.export.replace('.csv', '_comparison.csv')
                comparison.to_csv(export_path, index=False)
                print(f"\n✓ Comparison exported to {export_path}")
        
        return
    
    # === NON-TEMPORAL ANALYSIS ===
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Ensure text column exists
    if args.text_col not in df.columns:
        print(f"Error: Column '{args.text_col}' not found. Available columns: {list(df.columns)}")
        return
    
    # Rename for consistency
    if args.text_col != 'text':
        df['text'] = df[args.text_col]
    
    # STANCE ANALYSIS
    if args.stance:
        print("\n" + "=" * 80)
        print(f"STANCE ANALYSIS: {args.entity}")
        print("=" * 80)
        
        stances = analyze_entity_stance_distribution(df, args.entity, 'text')
        total = sum(stances.values())
        
        if total > 0:
            print(f"\nStance distribution (n={total}):")
            print(f"  Pro:     {stances['pro']:4d} ({stances['pro']/total*100:5.1f}%)")
            print(f"  Anti:    {stances['anti']:4d} ({stances['anti']/total*100:5.1f}%)")
            print(f"  Neutral: {stances['neutral']:4d} ({stances['neutral']/total*100:5.1f}%)")
            
            # Calculate stance score (-1 to +1)
            stance_score = (stances['pro'] - stances['anti']) / total if total > 0 else 0
            print(f"\nOverall stance score: {stance_score:+.3f}")
            print(f"  (-1.0 = fully anti, +1.0 = fully pro, 0.0 = neutral)")
        else:
            print(f"\nNo mentions of '{args.entity}' found in dataset.")
    
    # FRAMING ANALYSIS
    if args.framing:
        print("\n" + "=" * 80)
        print(f"FRAMING ANALYSIS: {args.entity}")
        print("=" * 80)
        
        analyzer = EntityFramingAnalyzer()
        
        # Filter to texts mentioning entity
        relevant_texts = df[df['text'].str.contains(args.entity, case=False, na=False)]['text'].tolist()
        
        if relevant_texts:
            print(f"\nAnalyzing {len(relevant_texts)} texts mentioning '{args.entity}'...")
            framing = analyzer.aggregate_framing(relevant_texts, args.entity, top_n=10)
            
            if not framing.empty:
                print("\nTop descriptors:")
                
                adj_dict = framing['adjectives'].iloc[0] if not framing.empty else {}
                if adj_dict:
                    print("\n  Adjectives:")
                    for word, count in list(adj_dict.items())[:10]:
                        print(f"    {word}: {count}")
                
                verb_dict = framing['verbs'].iloc[0] if not framing.empty else {}
                if verb_dict:
                    print("\n  Verbs:")
                    for word, count in list(verb_dict.items())[:10]:
                        print(f"    {word}: {count}")
                
                compound_dict = framing['compounds'].iloc[0] if not framing.empty else {}
                if compound_dict:
                    print("\n  Compounds:")
                    for phrase, count in list(compound_dict.items())[:10]:
                        print(f"    {phrase}: {count}")
            else:
                print("\nNo framing patterns detected.")
        else:
            print(f"\nNo mentions of '{args.entity}' found in dataset.")
    
    # GROUP COMPARISON
    if args.group_by:
        print("\n" + "=" * 80)
        print(f"SENTIMENT BY GROUP: {args.entity}")
        print("=" * 80)
        
        if args.group_by not in df.columns:
            print(f"Error: Group column '{args.group_by}' not found.")
            return
        
        comparison = compare_group_sentiment(df, args.entity, args.group_by, 'text')
        
        if not comparison.empty:
            print(f"\nSentiment toward '{args.entity}' by {args.group_by}:")
            print(comparison.to_string(index=False))
            
            # Highlight extremes
            if len(comparison) > 1:
                most_positive = comparison.iloc[0]
                most_negative = comparison.iloc[-1]
                
                print(f"\nMost positive: {most_positive[args.group_by]} "
                      f"(sentiment: {most_positive['mean_sentiment']:.3f}, "
                      f"n={most_positive['n_mentions']})")
                print(f"Most negative: {most_negative[args.group_by]} "
                      f"(sentiment: {most_negative['mean_sentiment']:.3f}, "
                      f"n={most_negative['n_mentions']})")
            
            if args.export:
                comparison.to_csv(args.export, index=False)
                print(f"\n✓ Group comparison exported to {args.export}")
        else:
            print(f"\nNo mentions of '{args.entity}' found with group '{args.group_by}'.")


if __name__ == "__main__":
    main()
