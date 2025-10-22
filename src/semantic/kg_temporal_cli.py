"""
Command-line interface for temporal knowledge graph analysis.

Usage:
    python -m src.semantic.kg_temporal_cli --input output/temporal --entity "China"
    
Author: AI Assistant
Date: 2025-10-21
"""

import argparse
from pathlib import Path
from src.semantic.kg_temporal import TemporalKG
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Analyze knowledge graphs across time periods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build entity timeline
  python -m src.semantic.kg_temporal_cli --input output/temporal --timeline
  
  # Analyze specific entity
  python -m src.semantic.kg_temporal_cli --input output/temporal --entity "China"
  
  # Detect events for entity
  python -m src.semantic.kg_temporal_cli --input output/temporal --entity "CIA" --events
  
  # Compare two periods
  python -m src.semantic.kg_temporal_cli --input output/temporal --compare 2023-01-15 2023-01-16
  
  # Generate full report
  python -m src.semantic.kg_temporal_cli --input output/temporal --report timeline_report.md
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Directory containing time-period subdirectories (from kg_cli.py --group-by-time)'
    )
    
    parser.add_argument(
        '--timeline',
        action='store_true',
        help='Build and display entity timeline'
    )
    
    parser.add_argument(
        '--entity',
        type=str,
        help='Analyze specific entity (shows trajectory, events, neighbors)'
    )
    
    parser.add_argument(
        '--events',
        action='store_true',
        help='Detect events for specified entity (requires --entity)'
    )
    
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('PERIOD1', 'PERIOD2'),
        help='Compare two time periods'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        metavar='OUTPUT.md',
        help='Generate comprehensive timeline report'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top entities to show (default: 20)'
    )
    
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for event detection (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Initialize TemporalKG
    print(f"Loading temporal KG from {args.input}...")
    tkg = TemporalKG(args.input)
    
    # Build timeline (needed for most operations)
    if args.timeline or args.entity or args.report or args.events:
        print("\nBuilding entity timeline...")
        timeline = tkg.build_entity_timeline()
        
        if args.timeline:
            print("\n" + "=" * 80)
            print("ENTITY TIMELINE")
            print("=" * 80)
            print(f"\nTotal entities tracked: {len(timeline)}")
            print(f"\nTop {args.top_n} entities by total mentions:")
            print(timeline.head(args.top_n).to_string(index=False))
            
            # Persistence distribution
            print("\n" + "-" * 80)
            print("Persistence Distribution:")
            print(f"  Highly persistent (>80%): {len(timeline[timeline['persistence'] > 0.8])}")
            print(f"  Moderately persistent (50-80%): {len(timeline[(timeline['persistence'] > 0.5) & (timeline['persistence'] <= 0.8)])}")
            print(f"  Low persistence (<50%): {len(timeline[timeline['persistence'] <= 0.5])}")
    
    # Entity-specific analysis
    if args.entity:
        entity_name = args.entity
        print("\n" + "=" * 80)
        print(f"ENTITY ANALYSIS: {entity_name}")
        print("=" * 80)
        
        # Get entity info from timeline
        if timeline is None:
            timeline = tkg.build_entity_timeline()
        
        entity_row = timeline[timeline['entity'] == entity_name]
        if entity_row.empty:
            print(f"Error: Entity '{entity_name}' not found in knowledge graph.")
            return
        
        entity_row = entity_row.iloc[0]
        
        print(f"\nBasic Info:")
        print(f"  Type: {entity_row['type']}")
        print(f"  First seen: {entity_row['first_seen']}")
        print(f"  Last seen: {entity_row['last_seen']}")
        print(f"  Lifespan: {entity_row['lifespan']} / {entity_row['total_periods']} periods")
        print(f"  Persistence: {entity_row['persistence']:.1%}")
        print(f"  Total mentions: {entity_row['total_mentions']}")
        print(f"  Peak: {entity_row['peak_frequency']} mentions in {entity_row['peak_period']}")
        
        # Trajectory classification
        trajectory_type = tkg.classify_trajectory(entity_name)
        print(f"  Trajectory: {trajectory_type}")
        
        # Show trajectory
        print(f"\nFrequency by period:")
        for period, freq in zip(tkg.periods, entity_row['trajectory']):
            bar = "█" * int(freq / max(entity_row['trajectory']) * 40) if max(entity_row['trajectory']) > 0 else ""
            print(f"  {period}: {freq:3d} {bar}")
        
        # Co-occurring entities
        print(f"\nTop co-occurring entities:")
        neighbors = tkg.get_entity_neighbors_over_time(entity_name)
        if not neighbors.empty:
            print(neighbors.head(10).to_string(index=False))
        else:
            print("  No co-occurrences found")
        
        # Event detection
        if args.events or True:  # Always show events for entity analysis
            print(f"\nEvent detection (z-threshold={args.z_threshold}):")
            events = tkg.detect_events(entity_name, z_threshold=args.z_threshold)
            if events:
                for event in events:
                    print(f"  ⚡ {event['period']}: {event['frequency']} mentions "
                          f"(z-score={event['z_score']:.2f}, baseline={event['baseline']:.1f})")
            else:
                print("  No significant events detected")
    
    # Period comparison
    if args.compare:
        period1, period2 = args.compare
        print("\n" + "=" * 80)
        print(f"PERIOD COMPARISON: {period1} vs {period2}")
        print("=" * 80)
        
        comparison = tkg.compare_periods(period1, period2, top_n=args.top_n)
        
        print(f"\nNew entities in {period2}:")
        if not comparison['new_entities'].empty:
            print(comparison['new_entities'][['entity', 'type', 'frequency']].head(args.top_n).to_string(index=False))
        else:
            print("  None")
        
        print(f"\nLost entities from {period1}:")
        if not comparison['lost_entities'].empty:
            print(comparison['lost_entities'][['entity', 'type', 'frequency']].head(args.top_n).to_string(index=False))
        else:
            print("  None")
        
        print(f"\nGrowing entities:")
        if not comparison['growing'].empty:
            print(comparison['growing'].to_string(index=False))
        else:
            print("  None")
        
        print(f"\nDeclining entities:")
        if not comparison['declining'].empty:
            print(comparison['declining'].to_string(index=False))
        else:
            print("  None")
    
    # Generate report
    if args.report:
        print("\n" + "=" * 80)
        print("GENERATING TIMELINE REPORT")
        print("=" * 80)
        tkg.export_timeline_report(args.report)
        print(f"✓ Report saved to {args.report}")


if __name__ == "__main__":
    main()
