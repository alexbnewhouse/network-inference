#!/usr/bin/env python3
"""
End-to-end 4chan data analysis example.

This script demonstrates a complete workflow for analyzing 4chan data:
1. Load and prepare 4chan-formatted CSV
2. Build semantic network (topic co-occurrence)
3. Extract knowledge graph (entities and relationships)
4. Perform temporal analysis (entity evolution over time)
5. Compare boards (cross-community analysis)
6. Generate summary report

Usage:
    python examples/4chan_analysis_example.py examples/sample_4chan.csv output/4chan_analysis
"""

import sys
import pandas as pd
from pathlib import Path
import subprocess
import json
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def get_edge_columns(edges_df):
    """
    Detect edge column names and return standardized tuple.
    
    Returns:
        tuple: (src_col, dst_col, weight_col)
    """
    if 'similarity' in edges_df.columns:
        return ('source', 'target', 'similarity')
    else:
        return ('src', 'dst', 'weight')


def get_entity_type_column(nodes_df):
    """
    Detect entity type column name.
    
    Returns:
        str: Column name for entity type ('entity_type' or 'type')
    """
    return 'entity_type' if 'entity_type' in nodes_df.columns else 'type'


def format_table_markdown(df):
    """
    Format DataFrame as markdown table.
    
    Args:
        df: DataFrame to format
        
    Returns:
        str: Markdown table string
    """
    lines = []
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n▶ {description}...")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Success")
        return True
    else:
        print(f"  ✗ Failed")
        print(f"  Error: {result.stderr[:500]}")
        return False


def analyze_4chan_data(input_file, output_dir):
    """
    Complete 4chan data analysis workflow.
    
    Args:
        input_file: Path to 4chan CSV file
        output_dir: Directory for output files
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print_section("4chan Data Analysis - Complete Workflow")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Step 0: Load and inspect data
    print_section("Step 0: Data Inspection")
    df = pd.read_csv(input_path)
    
    print(f"\nDataset overview:")
    print(f"  Total posts: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    if 'board' in df.columns:
        print(f"  Boards: {df['board'].value_counts().to_dict()}")
    
    if 'created_at' in df.columns or 'time' in df.columns:
        time_col = 'created_at' if 'created_at' in df.columns else 'time'
        if time_col == 'time':
            df['created_at'] = pd.to_datetime(df['time'], unit='s')
        else:
            df['created_at'] = pd.to_datetime(df['created_at'])
        print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    if 'tripcode' in df.columns:
        tripcode_pct = df['tripcode'].notna().sum() / len(df) * 100
        print(f"  Posts with tripcodes: {df['tripcode'].notna().sum()} ({tripcode_pct:.1f}%)")
    
    print(f"\nSample posts:")
    print(df[['board', 'body']].head(3).to_string(index=False))
    
    # Step 1: Semantic Network
    print_section("Step 1: Semantic Network Analysis")
    semantic_dir = output_path / "semantic"
    
    success = run_command([
        "python", "-m", "src.semantic.build_semantic_network",
        "--input", str(input_path),
        "--text-col", "body",
        "--outdir", str(semantic_dir),
        "--min-df", "3",
        "--topk", "20"
    ], "Building semantic network (topic co-occurrence)")
    
    if success and (semantic_dir / "edges.csv").exists():
        edges = pd.read_csv(semantic_dir / "edges.csv")
        print(f"\n  Network statistics:")
        print(f"    Edges: {len(edges)}")
        print(f"    Top connections:")
        src_col, dst_col, weight_col = get_edge_columns(edges)
        cols = [src_col, dst_col, weight_col]
        print(edges.nlargest(5, weight_col)[cols].to_string(index=False))
    
    # Step 2: Knowledge Graph
    print_section("Step 2: Knowledge Graph Extraction")
    kg_dir = output_path / "knowledge_graph"
    
    success = run_command([
        "python", "-m", "src.semantic.kg_cli",
        "--input", str(input_path),
        "--text-col", "body",
        "--outdir", str(kg_dir),
        "--max-rows", "500"
    ], "Extracting entities and relationships")
    
    if success and (kg_dir / "kg_nodes.csv").exists():
        nodes = pd.read_csv(kg_dir / "kg_nodes.csv")
        print(f"\n  Entity statistics:")
        print(f"    Total entities: {len(nodes)}")
        
        type_col = get_entity_type_column(nodes)
        
        if type_col in nodes.columns:
            print(f"    By type:")
            for etype, count in nodes[type_col].value_counts().head(5).items():
                print(f"      {etype}: {count}")
        
        print(f"\n    Top entities:")
        cols = ['entity', type_col, 'frequency']
        print(nodes.nlargest(10, 'frequency')[cols].to_string(index=False))
    
    # Step 3: Temporal Analysis (if timestamps available)
    if 'created_at' in df.columns or 'time' in df.columns:
        print_section("Step 3: Temporal Knowledge Graph")
        temporal_dir = output_path / "temporal"
        
        # Determine time grouping based on date range
        date_range = (df['created_at'].max() - df['created_at'].min()).days
        if date_range > 60:
            time_freq = "monthly"
        elif date_range > 14:
            time_freq = "weekly"
        else:
            time_freq = "daily"
        
        print(f"\n  Date range: {date_range} days → Using {time_freq} grouping")
        
        success = run_command([
            "python", "-m", "src.semantic.kg_cli",
            "--input", str(input_path),
            "--text-col", "body",
            "--time-col", "created_at",
            "--group-by-time", time_freq,
            "--outdir", str(temporal_dir),
            "--max-rows", "500"
        ], f"Building temporal knowledge graphs ({time_freq})")
        
        if success:
            # Count periods
            period_dirs = list(temporal_dir.glob("period_*"))
            print(f"\n  Generated {len(period_dirs)} time periods")
            
            # Analyze entity evolution
            if period_dirs:
                print(f"\n  Analyzing entity evolution across time periods...")
                
                # Find top entities overall
                all_entities = {}
                for period_dir in period_dirs:
                    nodes_file = period_dir / "kg_nodes.csv"
                    if nodes_file.exists():
                        period_nodes = pd.read_csv(nodes_file)
                        for _, row in period_nodes.iterrows():
                            entity = row['entity']
                            all_entities[entity] = all_entities.get(entity, 0) + row['frequency']
                
                # Track top 3 entities
                top_entities = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"\n  Top entities overall:")
                for entity, freq in top_entities:
                    print(f"    {entity}: {freq} mentions")
                
                # Track first entity over time
                if top_entities:
                    tracked_entity = top_entities[0][0]
                    print(f"\n  Timeline for '{tracked_entity}':")
                    
                    for period_dir in sorted(period_dirs):
                        period = period_dir.name.replace("period_", "")
                        nodes_file = period_dir / "kg_nodes.csv"
                        
                        if nodes_file.exists():
                            period_nodes = pd.read_csv(nodes_file)
                            entity_row = period_nodes[period_nodes['entity'] == tracked_entity]
                            
                            if not entity_row.empty:
                                freq = entity_row.iloc[0]['frequency']
                                print(f"    {period}: {freq} mentions")
    
    # Step 4: Board Comparison (if board column exists)
    if 'board' in df.columns:
        print_section("Step 4: Board-Level Comparison")
        
        boards = df['board'].unique()
        print(f"\n  Analyzing {len(boards)} boards: {list(boards)}")
        
        board_stats = []
        
        for board in boards[:3]:  # Limit to first 3 boards for demo
            board_dir = output_path / f"board_{board}"
            
            # Filter data for this board
            board_data_file = output_path / f"board_{board}_data.csv"
            df_board = df[df['board'] == board]
            df_board.to_csv(board_data_file, index=False)
            
            success = run_command([
                "python", "-m", "src.semantic.kg_cli",
                "--input", str(board_data_file),
                "--text-col", "body",
                "--outdir", str(board_dir),
                "--max-rows", "500"
            ], f"Analyzing /{board}/")
            
            if success and (board_dir / "kg_nodes.csv").exists():
                nodes = pd.read_csv(board_dir / "kg_nodes.csv")
                type_col = get_entity_type_column(nodes)
                board_stats.append({
                    'board': board,
                    'posts': len(df_board),
                    'unique_entities': len(nodes),
                    'person_mentions': len(nodes[nodes[type_col] == 'PERSON']) if type_col in nodes.columns else 0,
                    'org_mentions': len(nodes[nodes[type_col] == 'ORG']) if type_col in nodes.columns else 0,
                })
        
        if board_stats:
            print(f"\n  Board comparison:")
            df_stats = pd.DataFrame(board_stats)
            print(df_stats.to_string(index=False))
    
    # Step 5: Generate Summary Report
    print_section("Step 5: Summary Report")
    
    report_path = output_path / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# 4chan Data Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Input File:** `{input_path}`\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total posts:** {len(df)}\n")
        if 'board' in df.columns:
            f.write(f"- **Boards:** {', '.join(df['board'].unique())}\n")
        if 'created_at' in df.columns:
            f.write(f"- **Date range:** {df['created_at'].min()} to {df['created_at'].max()}\n")
        
        f.write("\n## Analysis Outputs\n\n")
        
        if (semantic_dir / "edges.csv").exists():
            edges = pd.read_csv(semantic_dir / "edges.csv")
            f.write(f"### Semantic Network\n\n")
            f.write(f"- **Location:** `{semantic_dir.relative_to(output_path)}/`\n")
            f.write(f"- **Edges:** {len(edges)}\n")
            f.write(f"- **Top connections:**\n\n")
            src_col, dst_col, weight_col = get_edge_columns(edges)
            top_edges = edges.nlargest(10, weight_col)
            for _, row in top_edges.iterrows():
                f.write(f"  - {row[src_col]} ↔ {row[dst_col]} ({weight_col}: {row[weight_col]:.2f})\n")
        
        if (kg_dir / "kg_nodes.csv").exists():
            nodes = pd.read_csv(kg_dir / "kg_nodes.csv")
            f.write(f"\n### Knowledge Graph\n\n")
            f.write(f"- **Location:** `{kg_dir.relative_to(output_path)}/`\n")
            f.write(f"- **Entities:** {len(nodes)}\n")
            f.write(f"- **Top entities:**\n\n")
            type_col = get_entity_type_column(nodes)
            top_entities = nodes.nlargest(15, 'frequency')
            for _, row in top_entities.iterrows():
                etype = row.get(type_col, 'UNKNOWN')
                f.write(f"  - {row['entity']} ({etype}): {row['frequency']} mentions\n")
        
        if board_stats:
            f.write(f"\n### Board Comparison\n\n")
            df_stats = pd.DataFrame(board_stats)
            f.write(format_table_markdown(df_stats) + "\n")
        
        f.write(f"\n\n## Files Generated\n\n")
        f.write(f"- Semantic network: `{semantic_dir.relative_to(output_path)}/`\n")
        f.write(f"- Knowledge graph: `{kg_dir.relative_to(output_path)}/`\n")
        if temporal_dir.exists():
            f.write(f"- Temporal analysis: `{temporal_dir.relative_to(output_path)}/`\n")
        
        f.write(f"\n## Next Steps\n\n")
        f.write("1. **Visualize networks:** Open `.graphml` files in Gephi\n")
        f.write("2. **Analyze entities:** Review `kg_nodes.csv` for entity patterns\n")
        f.write("3. **Track evolution:** Explore temporal directories for time-series analysis\n")
        f.write("4. **Compare boards:** Review board-specific outputs\n")
        
        f.write(f"\n---\n\n")
        f.write(f"*Analysis completed with Network Inference Toolkit*\n")
    
    print(f"\n  ✓ Report saved to: {report_path}")
    
    # Final summary
    print_section("Analysis Complete!")
    print(f"\n✓ All outputs saved to: {output_path}")
    print(f"\n📊 Key findings:")
    print(f"   - Processed {len(df)} posts")
    
    if (kg_dir / "kg_nodes.csv").exists():
        nodes = pd.read_csv(kg_dir / "kg_nodes.csv")
        print(f"   - Extracted {len(nodes)} unique entities")
    
    if (semantic_dir / "edges.csv").exists():
        edges = pd.read_csv(semantic_dir / "edges.csv")
        print(f"   - Built semantic network with {len(edges)} topic connections")
    
    print(f"\n📄 Read full report: {report_path}")
    print(f"\n🎨 Visualize: Open *.graphml files in Gephi")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python 4chan_analysis_example.py <input_csv> [output_dir]")
        print("\nExample:")
        print("  python examples/4chan_analysis_example.py examples/sample_4chan.csv output/analysis")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/4chan_analysis"
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    analyze_4chan_data(input_file, output_dir)


if __name__ == "__main__":
    main()
