"""
CLI for user-entity network analysis.

Usage examples:
    # Build network and get stats
    python -m src.semantic.kg_user_entity_network_cli --kg-dir output/kg_quickwins --data pol_archive_0.csv --user-col user_id --text-col body --stats
    
    # Analyze user profile
    python -m src.semantic.kg_user_entity_network_cli --kg-dir output/kg_quickwins --data pol_archive_0.csv --user-col user_id --text-col body --user USER123
    
    # Find similar users
    python -m src.semantic.kg_user_entity_network_cli --kg-dir output/kg_quickwins --data pol_archive_0.csv --user-col user_id --text-col body --similar-users USER123
    
    # Export networks
    python -m src.semantic.kg_user_entity_network_cli --kg-dir output/kg_quickwins --data pol_archive_0.csv --user-col user_id --text-col body --export-all output/networks

Author: AI Assistant
Date: 2025-10-21
"""

import argparse
import pandas as pd
from pathlib import Path
from src.semantic.kg_user_entity_network import UserEntityNetwork, load_from_kg_output


def main():
    parser = argparse.ArgumentParser(
        description="User-Entity Network Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument('--kg-dir', required=True, help='KG output directory with kg_nodes.csv')
    parser.add_argument('--data', required=True, help='Original data CSV file')
    parser.add_argument('--user-col', required=True, help='User ID column name')
    parser.add_argument('--text-col', required=True, help='Text column name')
    
    # Analysis options
    parser.add_argument('--stats', action='store_true', help='Show network statistics')
    parser.add_argument('--user', help='Analyze specific user profile')
    parser.add_argument('--entity', help='Analyze specific entity audience')
    parser.add_argument('--similar-users', metavar='USER_ID', help='Find similar users')
    parser.add_argument('--related-entities', metavar='ENTITY', help='Find related entities')
    parser.add_argument('--communities', action='store_true', help='Detect user communities')
    
    # Export options
    parser.add_argument('--export-bipartite', help='Export bipartite graph to GraphML')
    parser.add_argument('--export-user-network', help='Export user-user network to GraphML')
    parser.add_argument('--export-entity-network', help='Export entity-entity network to GraphML')
    parser.add_argument('--export-matrix', help='Export user-entity matrix to CSV')
    parser.add_argument('--export-all', help='Export all networks to directory')
    
    # Parameters
    parser.add_argument('--top-n', type=int, default=10, help='Number of top results to show')
    parser.add_argument('--similarity', default='jaccard', choices=['jaccard', 'cosine', 'overlap'],
                       help='Similarity metric for user/entity comparison')
    parser.add_argument('--min-shared', type=int, default=2, help='Minimum shared entities/users for network projection')
    
    args = parser.parse_args()
    
    # Load network
    print(f"Loading user-entity network from {args.kg_dir}...")
    try:
        network = load_from_kg_output(
            kg_dir=args.kg_dir,
            data_path=args.data,
            user_col=args.user_col,
            text_col=args.text_col
        )
    except Exception as e:
        print(f"Error loading network: {e}")
        return
    
    print(f"✓ Loaded network: {len(network.users)} users, {len(network.entities)} entities")
    
    # STATISTICS
    if args.stats:
        print("\n" + "=" * 80)
        print("NETWORK STATISTICS")
        print("=" * 80)
        
        stats = network.get_stats()
        print(f"\nNodes:")
        print(f"  Users:    {stats['n_users']:,}")
        print(f"  Entities: {stats['n_entities']:,}")
        print(f"\nEdges: {stats['n_edges']:,}")
        print(f"\nAverage mentions:")
        print(f"  Entities per user: {stats['avg_entities_per_user']:.1f}")
        print(f"  Users per entity:  {stats['avg_users_per_entity']:.1f}")
        print(f"\nDensity: {stats['density']:.6f}")
        
        # Top entities by user count
        entity_user_counts = [(e, len(network.entity_users[e])) for e in network.entities]
        entity_user_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop entities by user count:")
        for entity, n_users in entity_user_counts[:10]:
            print(f"  {entity}: {n_users} users")
        
        # Top users by entity count
        user_entity_counts = [(u, len(network.user_entities[u])) for u in network.users]
        user_entity_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop users by entity count:")
        for user_id, n_entities in user_entity_counts[:10]:
            user_display = user_id[:20] + "..." if len(user_id) > 20 else user_id
            print(f"  {user_display}: {n_entities} entities")
    
    # USER PROFILE
    if args.user:
        print("\n" + "=" * 80)
        print(f"USER PROFILE: {args.user}")
        print("=" * 80)
        
        profile = network.get_user_profile(args.user, top_n=args.top_n)
        
        if profile.empty:
            print(f"\nUser '{args.user}' not found in network.")
        else:
            print(f"\nTop {len(profile)} entities mentioned:")
            print(profile.to_string(index=False))
            
            total_mentions = profile['mention_count'].sum()
            total_entities = len(network.user_entities[args.user])
            print(f"\nTotal: {total_entities} unique entities, {total_mentions} mentions")
    
    # ENTITY AUDIENCE
    if args.entity:
        print("\n" + "=" * 80)
        print(f"ENTITY AUDIENCE: {args.entity}")
        print("=" * 80)
        
        audience = network.get_entity_audience(args.entity, top_n=args.top_n)
        
        if audience.empty:
            print(f"\nEntity '{args.entity}' not found in network.")
        else:
            print(f"\nTop {len(audience)} users mentioning this entity:")
            for _, row in audience.iterrows():
                user_display = row['user_id'][:30] + "..." if len(row['user_id']) > 30 else row['user_id']
                print(f"  {user_display}: {row['mention_count']} mentions")
            
            total_users = len(network.entity_users[args.entity])
            total_mentions = audience['mention_count'].sum()
            print(f"\nTotal: {total_users} unique users, {total_mentions} mentions (from top {len(audience)})")
    
    # SIMILAR USERS
    if args.similar_users:
        print("\n" + "=" * 80)
        print(f"SIMILAR USERS: {args.similar_users}")
        print("=" * 80)
        
        similar = network.find_similar_users(
            args.similar_users, 
            top_n=args.top_n,
            method=args.similarity
        )
        
        if similar.empty:
            print(f"\nNo similar users found for '{args.similar_users}'.")
        else:
            print(f"\nTop {len(similar)} similar users ({args.similarity} similarity):")
            print(similar.to_string(index=False))
    
    # RELATED ENTITIES
    if args.related_entities:
        print("\n" + "=" * 80)
        print(f"RELATED ENTITIES: {args.related_entities}")
        print("=" * 80)
        
        related = network.find_related_entities(
            args.related_entities,
            top_n=args.top_n,
            method=args.similarity
        )
        
        if related.empty:
            print(f"\nNo related entities found for '{args.related_entities}'.")
        else:
            print(f"\nTop {len(related)} related entities ({args.similarity} similarity):")
            print(related.to_string(index=False))
    
    # COMMUNITY DETECTION
    if args.communities:
        print("\n" + "=" * 80)
        print("USER COMMUNITIES")
        print("=" * 80)
        
        print("\nDetecting communities (this may take a moment)...")
        communities = network.detect_user_communities(method='label_prop')
        
        if not communities:
            print("\nNo communities detected.")
        else:
            # Count users per community
            from collections import Counter
            comm_counts = Counter(communities.values())
            
            print(f"\nDetected {len(comm_counts)} communities:")
            for comm_id, count in comm_counts.most_common():
                print(f"  Community {comm_id}: {count} users")
            
            # Show sample entities for top communities
            print("\nTop entities per community:")
            for comm_id, _ in list(comm_counts.most_common(5)):
                # Get users in this community
                comm_users = [u for u, c in communities.items() if c == comm_id]
                
                # Aggregate entities mentioned by community
                entity_counts = {}
                for user_id in comm_users:
                    for entity in network.user_entities[user_id]:
                        entity_counts[entity] = entity_counts.get(entity, 0) + 1
                
                top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                entities_str = ", ".join([f"{e} ({c})" for e, c in top_entities])
                print(f"\n  Community {comm_id}: {entities_str}")
    
    # EXPORT OPTIONS
    export_dir = Path(args.export_all) if args.export_all else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
    
    if args.export_bipartite or export_dir:
        path = export_dir / "bipartite_graph.graphml" if export_dir else args.export_bipartite
        print(f"\nExporting bipartite graph to {path}...")
        network.export_bipartite_graph(str(path))
        print("✓ Done")
    
    if args.export_user_network or export_dir:
        path = export_dir / "user_network.graphml" if export_dir else args.export_user_network
        print(f"\nExporting user-user network to {path}...")
        network.export_user_network(str(path), min_shared_entities=args.min_shared)
        print("✓ Done")
    
    if args.export_entity_network or export_dir:
        path = export_dir / "entity_network.graphml" if export_dir else args.export_entity_network
        print(f"\nExporting entity-entity network to {path}...")
        network.export_entity_network(str(path), min_shared_users=args.min_shared)
        print("✓ Done")
    
    if args.export_matrix or export_dir:
        path = export_dir / "user_entity_matrix.csv" if export_dir else args.export_matrix
        print(f"\nExporting user-entity matrix to {path}...")
        network.export_user_entity_matrix(str(path))
        print("✓ Done")


if __name__ == "__main__":
    main()
