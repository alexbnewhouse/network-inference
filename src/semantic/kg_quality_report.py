"""
Knowledge Graph Quality Report Generator
"""
import pandas as pd
import os
from collections import Counter


def generate_kg_report(nodes_df, edges_df, outdir, df_original=None):
    """
    Generate a comprehensive quality report for extracted knowledge graph.
    
    Args:
        nodes_df: DataFrame of KG nodes
        edges_df: DataFrame of KG edges
        outdir: Output directory
        df_original: Optional original input DataFrame for additional stats
    """
    report_lines = []
    report_lines.append("# Knowledge Graph Quality Report\n")
    report_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")
    
    # === BASIC STATISTICS ===
    report_lines.append("## Basic Statistics\n")
    report_lines.append(f"- **Total Entities**: {len(nodes_df)}\n")
    report_lines.append(f"- **Total Relationships**: {len(edges_df)}\n")
    report_lines.append(f"- **Average Degree**: {(2 * len(edges_df) / len(nodes_df)):.2f}\n")
    
    if df_original is not None:
        report_lines.append(f"- **Input Documents**: {len(df_original)}\n")
        report_lines.append(f"- **Entities per Document**: {len(nodes_df) / len(df_original):.2f}\n")
    report_lines.append("\n")
    
    # === ENTITY TYPE DISTRIBUTION ===
    report_lines.append("## Entity Type Distribution\n")
    type_counts = nodes_df['type'].value_counts()
    report_lines.append("| Entity Type | Count | Percentage |\n")
    report_lines.append("|-------------|-------|------------|\n")
    for etype, count in type_counts.items():
        pct = 100 * count / len(nodes_df)
        report_lines.append(f"| {etype} | {count} | {pct:.1f}% |\n")
    report_lines.append("\n")
    
    # === TOP ENTITIES ===
    report_lines.append("## Top 20 Most Frequent Entities\n")
    top_entities = nodes_df.nlargest(20, 'frequency')
    report_lines.append("| Rank | Entity | Type | Frequency |\n")
    report_lines.append("|------|--------|------|----------|\n")
    for i, row in enumerate(top_entities.itertuples(), 1):
        report_lines.append(f"| {i} | {row.entity} | {row.type} | {row.frequency} |\n")
    report_lines.append("\n")
    
    # === RELATION TYPES ===
    report_lines.append("## Relationship Types\n")
    if 'relation_type' in edges_df.columns:
        rel_counts = edges_df['relation_type'].value_counts()
        report_lines.append("| Relation Type | Count | Percentage |\n")
        report_lines.append("|--------------|-------|------------|\n")
        for rel_type, count in rel_counts.items():
            pct = 100 * count / len(edges_df)
            report_lines.append(f"| {rel_type} | {count} | {pct:.1f}% |\n")
        report_lines.append("\n")
        
        # Top dependency relations if available
        dep_edges = edges_df[edges_df['relation_type'] != 'co-occurrence']
        if len(dep_edges) > 0:
            report_lines.append("### Sample Dependency-Based Relations\n")
            sample_deps = dep_edges.head(10)
            report_lines.append("| Source | Relation | Target |\n")
            report_lines.append("|--------|----------|--------|\n")
            for row in sample_deps.itertuples():
                report_lines.append(f"| {row.source_entity} | {row.relation_type} | {row.target_entity} |\n")
            report_lines.append("\n")
    
    # === STRONGEST CO-OCCURRENCES ===
    if 'weight' in edges_df.columns:
        report_lines.append("## Top 15 Strongest Entity Co-occurrences\n")
        cooc = edges_df[edges_df['relation_type'] == 'co-occurrence'].nlargest(15, 'weight')
        report_lines.append("| Source | Target | Weight |\n")
        report_lines.append("|--------|--------|--------|\n")
        for row in cooc.itertuples():
            report_lines.append(f"| {row.source_entity} | {row.target_entity} | {row.weight} |\n")
        report_lines.append("\n")
    
    # === QUALITY CHECKS ===
    report_lines.append("## Quality Checks\n")
    
    # Check for self-loops
    self_loops = edges_df[edges_df['source_entity'] == edges_df['target_entity']]
    if len(self_loops) > 0:
        report_lines.append(f"⚠️  **Warning**: Found {len(self_loops)} self-loop edges\n")
    else:
        report_lines.append("✓ No self-loops detected\n")
    
    # Check for case duplicates
    entity_lower = nodes_df['entity'].str.lower()
    duplicates = entity_lower[entity_lower.duplicated(keep=False)]
    if len(duplicates) > 0:
        report_lines.append(f"⚠️  **Warning**: Found {len(duplicates)} potential case duplicates\n")
        report_lines.append("  Example duplicates:\n")
        for ent_lower in duplicates.unique()[:5]:
            variants = nodes_df[entity_lower == ent_lower]['entity'].tolist()
            report_lines.append(f"  - {variants}\n")
    else:
        report_lines.append("✓ No case duplicates detected\n")
    
    # Check entity type diversity
    if len(type_counts) < 3:
        report_lines.append(f"⚠️  **Warning**: Low entity type diversity ({len(type_counts)} types)\n")
    else:
        report_lines.append(f"✓ Good entity type diversity ({len(type_counts)} types)\n")
    
    # Check if one type dominates
    if len(type_counts) > 0:
        top_type_pct = 100 * type_counts.iloc[0] / len(nodes_df)
        if top_type_pct > 70:
            report_lines.append(f"⚠️  **Warning**: One entity type dominates ({type_counts.index[0]}: {top_type_pct:.1f}%)\n")
        else:
            report_lines.append(f"✓ Balanced entity type distribution\n")
    
    # Check extraction rate
    if df_original is not None:
        entities_per_doc = len(nodes_df) / len(df_original)
        if entities_per_doc < 0.1:
            report_lines.append(f"⚠️  **Warning**: Very low entity extraction rate ({entities_per_doc:.3f} entities/doc)\n")
            report_lines.append("  Consider: lowering --min-freq, checking data quality, or using better NER model\n")
        elif entities_per_doc > 10:
            report_lines.append(f"⚠️  **Note**: High entity extraction rate ({entities_per_doc:.2f} entities/doc)\n")
            report_lines.append("  Consider: raising --min-freq to focus on more important entities\n")
        else:
            report_lines.append(f"✓ Reasonable entity extraction rate ({entities_per_doc:.2f} entities/doc)\n")
    
    report_lines.append("\n")
    
    # === METADATA SUMMARY ===
    if 'first_context' in nodes_df.columns:
        report_lines.append("## Entity Metadata\n")
        report_lines.append(f"✓ First context samples available for {nodes_df['first_context'].notna().sum()} entities\n")
        
    if 'n_unique_contexts' in nodes_df.columns:
        report_lines.append(f"✓ Context diversity tracked ({nodes_df['n_unique_contexts'].mean():.1f} contexts/entity avg)\n")
    
    report_lines.append("\n")
    
    # === RECOMMENDATIONS ===
    report_lines.append("## Recommendations\n")
    
    if len(edges_df) / len(nodes_df) < 2:
        report_lines.append("- Graph is sparsely connected. Consider:\n")
        report_lines.append("  - Increasing --window size for more co-occurrences\n")
        report_lines.append("  - Lowering --min-freq to include more entities\n")
    
    if 'relation_type' in edges_df.columns:
        dep_count = len(edges_df[edges_df['relation_type'] != 'co-occurrence'])
        if dep_count < len(edges_df) * 0.05:
            report_lines.append("- Very few dependency-based relations extracted. Consider:\n")
            report_lines.append("  - Using better spaCy model (en_core_web_md or en_core_web_lg)\n")
            report_lines.append("  - Checking that text is well-formed (punctuation, grammar)\n")
    
    report_lines.append("\n---\n")
    report_lines.append("*Report generated by Knowledge Graph Pipeline*\n")
    
    # Write report
    report_path = os.path.join(outdir, "kg_quality_report.md")
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"✓ Quality report saved to {report_path}")
    return report_path
