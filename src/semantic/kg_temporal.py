"""
Temporal Knowledge Graph Analysis

This module provides tools for analyzing knowledge graphs over time:
- Entity lifecycle tracking (birth, peak, death)
- Event detection (sudden spikes in entity mentions)
- Entity trajectory analysis (growth, decline, stability)
- Cross-period comparisons

Author: AI Assistant
Date: 2025-10-21
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime


class TemporalKG:
    """
    Analyze knowledge graphs across multiple time periods.
    
    This class loads KG outputs from time-sliced directories (created by kg_cli.py
    with --group-by-time) and provides temporal analysis methods.
    
    Attributes:
        base_dir: Root directory containing time-period subdirectories
        periods: List of time period identifiers (e.g., ['2023-01-15', '2023-01-16'])
        nodes_data: Dict mapping period -> DataFrame of nodes
        edges_data: Dict mapping period -> DataFrame of edges
        entity_timeline: DataFrame with entity lifecycles
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize TemporalKG from a directory of time-sliced KG outputs.
        
        Args:
            base_dir: Path to directory containing subdirectories named by time period
                     (e.g., "output/temporal/" containing "2023-01-15/", "2023-01-16/", etc.)
        """
        self.base_dir = Path(base_dir)
        self.periods: List[str] = []
        self.nodes_data: Dict[str, pd.DataFrame] = {}
        self.edges_data: Dict[str, pd.DataFrame] = {}
        self.entity_timeline: Optional[pd.DataFrame] = None
        
        self._load_periods()
        
    def _load_periods(self) -> None:
        """Discover and load all time period directories (handles nested structures)."""
        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")
        
        # Recursively find all directories with KG data
        for root, dirs, files in sorted(self.base_dir.walk()):
            if "kg_nodes.csv" in files and "kg_edges.csv" in files:
                nodes_file = root / "kg_nodes.csv"
                edges_file = root / "kg_edges.csv"
                
                # Use relative path as period name
                period = str(root.relative_to(self.base_dir))
                self.periods.append(period)
                self.nodes_data[period] = pd.read_csv(nodes_file)
                self.edges_data[period] = pd.read_csv(edges_file)
        
        # Sort periods chronologically
        self.periods = sorted(self.periods)
        
        if not self.periods:
            raise ValueError(f"No valid KG data found in {self.base_dir}")
        
        print(f"Loaded {len(self.periods)} time periods: {self.periods[0]} to {self.periods[-1]}")
    
    def build_entity_timeline(self) -> pd.DataFrame:
        """
        Build a comprehensive timeline for all entities across all periods.
        
        Returns:
            DataFrame with columns:
                - entity: Entity name
                - type: Entity type (PERSON, ORG, etc.)
                - first_seen: First time period where entity appears
                - last_seen: Last time period where entity appears
                - lifespan: Number of periods entity appears in
                - total_periods: Total number of periods in dataset
                - persistence: lifespan / total_periods (0-1)
                - peak_period: Period with highest frequency
                - peak_frequency: Frequency at peak
                - total_mentions: Sum of frequency across all periods
                - trajectory: List of frequencies by period
        """
        timeline_records = []
        
        # Collect all unique entities
        all_entities = set()
        for nodes_df in self.nodes_data.values():
            all_entities.update(nodes_df['entity'].values)
        
        # Build timeline for each entity
        for entity in all_entities:
            periods_present = []
            frequencies = []
            entity_type = None
            
            for period in self.periods:
                nodes_df = self.nodes_data[period]
                entity_row = nodes_df[nodes_df['entity'] == entity]
                
                if not entity_row.empty:
                    freq = entity_row['frequency'].values[0]
                    entity_type = entity_row['type'].values[0]
                    periods_present.append(period)
                    frequencies.append(freq)
                else:
                    frequencies.append(0)
            
            if periods_present:
                # Calculate metrics
                first_seen = periods_present[0]
                last_seen = periods_present[-1]
                lifespan = len(periods_present)
                total_periods = len(self.periods)
                persistence = lifespan / total_periods
                
                # Find peak
                peak_idx = np.argmax(frequencies)
                peak_period = self.periods[peak_idx]
                peak_frequency = frequencies[peak_idx]
                total_mentions = sum(frequencies)
                
                timeline_records.append({
                    'entity': entity,
                    'type': entity_type,
                    'first_seen': first_seen,
                    'last_seen': last_seen,
                    'lifespan': lifespan,
                    'total_periods': total_periods,
                    'persistence': persistence,
                    'peak_period': peak_period,
                    'peak_frequency': peak_frequency,
                    'total_mentions': total_mentions,
                    'trajectory': frequencies
                })
        
        self.entity_timeline = pd.DataFrame(timeline_records)
        self.entity_timeline = self.entity_timeline.sort_values('total_mentions', ascending=False)
        
        return self.entity_timeline
    
    def detect_events(
        self, 
        entity: str, 
        z_threshold: float = 2.0,
        window_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detect significant events (spikes) for a specific entity.
        
        Uses z-score detection: if frequency in a period is > z_threshold 
        standard deviations above the moving average, it's marked as an event.
        
        Args:
            entity: Entity name to analyze
            z_threshold: Z-score threshold for event detection (default: 2.0)
            window_size: Size of moving average window (default: 3)
        
        Returns:
            List of event dictionaries with keys:
                - period: Time period of event
                - frequency: Entity frequency during event
                - z_score: Z-score of the spike
                - baseline: Moving average before the spike
        """
        if self.entity_timeline is None:
            self.build_entity_timeline()
        
        # Get entity trajectory
        entity_row = self.entity_timeline[self.entity_timeline['entity'] == entity]
        if entity_row.empty:
            return []
        
        trajectory = entity_row.iloc[0]['trajectory']
        
        if len(trajectory) < window_size + 1:
            return []  # Not enough data
        
        events = []
        
        for i in range(window_size, len(trajectory)):
            # Calculate moving average and std from previous window
            window = trajectory[i - window_size:i]
            baseline = np.mean(window)
            std = np.std(window)
            
            if std == 0:
                continue  # Avoid division by zero
            
            current_freq = trajectory[i]
            z_score = (current_freq - baseline) / std
            
            if z_score > z_threshold:
                events.append({
                    'period': self.periods[i],
                    'frequency': current_freq,
                    'z_score': z_score,
                    'baseline': baseline
                })
        
        return events
    
    def classify_trajectory(self, entity: str) -> str:
        """
        Classify an entity's trajectory pattern.
        
        Patterns:
            - "emerging": Growing over time (positive trend)
            - "declining": Decreasing over time (negative trend)
            - "stable": Relatively constant
            - "spike": Single major peak
            - "episodic": Multiple peaks
            - "unknown": Insufficient data
        
        Args:
            entity: Entity name to classify
        
        Returns:
            Trajectory classification string
        """
        if self.entity_timeline is None:
            self.build_entity_timeline()
        
        entity_row = self.entity_timeline[self.entity_timeline['entity'] == entity]
        if entity_row.empty:
            return "unknown"
        
        trajectory = np.array(entity_row.iloc[0]['trajectory'])
        non_zero = trajectory[trajectory > 0]
        
        if len(non_zero) < 3:
            return "unknown"
        
        # Calculate trend (linear regression slope)
        x = np.arange(len(trajectory))
        slope, _ = np.polyfit(x, trajectory, 1)
        
        # Calculate coefficient of variation
        mean_freq = np.mean(non_zero)
        std_freq = np.std(non_zero)
        cv = std_freq / mean_freq if mean_freq > 0 else 0
        
        # Count peaks (values significantly above neighbors)
        peaks = 0
        for i in range(1, len(trajectory) - 1):
            if trajectory[i] > trajectory[i-1] and trajectory[i] > trajectory[i+1]:
                if trajectory[i] > mean_freq * 1.5:  # Significant peak
                    peaks += 1
        
        # Classification logic
        if cv < 0.3:  # Low variation
            return "stable"
        elif slope > 0.5:  # Strong positive trend
            return "emerging"
        elif slope < -0.5:  # Strong negative trend
            return "declining"
        elif peaks >= 3:
            return "episodic"
        elif peaks == 1:
            return "spike"
        else:
            return "stable"
    
    def get_entity_neighbors_over_time(self, entity: str) -> pd.DataFrame:
        """
        Get entities that co-occur with the target entity across time periods.
        
        Args:
            entity: Target entity name
        
        Returns:
            DataFrame with columns:
                - neighbor: Co-occurring entity
                - periods_together: List of periods where they co-occur
                - co_occurrence_count: Number of periods with co-occurrence
                - total_weight: Sum of edge weights across all periods
        """
        neighbor_records = []
        neighbor_dict: Dict[str, Dict[str, Any]] = {}
        
        for period in self.periods:
            edges_df = self.edges_data[period]
            nodes_df = self.nodes_data[period]
            
            # Get entity ID from nodes
            entity_row = nodes_df[nodes_df['entity'] == entity]
            if entity_row.empty:
                continue
            entity_id = entity_row.iloc[0]['id']
            
            # Find edges involving the entity ID
            entity_edges = edges_df[
                (edges_df['source'] == entity_id) | (edges_df['target'] == entity_id)
            ]
            
            for _, edge in entity_edges.iterrows():
                # Get the neighbor entity name
                neighbor_id = edge['target'] if edge['source'] == entity_id else edge['source']
                neighbor_name = edge['target_entity'] if edge['source'] == entity_id else edge['source_entity']
                weight = edge.get('weight', 1)
                
                if neighbor_name not in neighbor_dict:
                    neighbor_dict[neighbor_name] = {
                        'periods': [],
                        'total_weight': 0
                    }
                
                neighbor_dict[neighbor_name]['periods'].append(period)
                neighbor_dict[neighbor_name]['total_weight'] += weight
        
        # Convert to records
        for neighbor, data in neighbor_dict.items():
            neighbor_records.append({
                'neighbor': neighbor,
                'periods_together': data['periods'],
                'co_occurrence_count': len(data['periods']),
                'total_weight': data['total_weight']
            })
        
        df = pd.DataFrame(neighbor_records)
        if not df.empty:
            df = df.sort_values('total_weight', ascending=False)
        
        return df
    
    def compare_periods(
        self, 
        period1: str, 
        period2: str,
        top_n: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare two time periods to find differences in entities and relationships.
        
        Args:
            period1: First time period identifier
            period2: Second time period identifier
            top_n: Number of top entities to compare (default: 20)
        
        Returns:
            Dictionary with keys:
                - 'new_entities': Entities only in period2
                - 'lost_entities': Entities only in period1
                - 'growing': Entities with increased frequency
                - 'declining': Entities with decreased frequency
                - 'stable': Entities with similar frequency
        """
        if period1 not in self.periods or period2 not in self.periods:
            raise ValueError(f"Invalid period(s). Available: {self.periods}")
        
        # Load and aggregate nodes (group by entity in case of duplicates)
        nodes1 = self.nodes_data[period1].groupby('entity').agg({
            'frequency': 'sum',
            'type': 'first'
        })
        nodes2 = self.nodes_data[period2].groupby('entity').agg({
            'frequency': 'sum',
            'type': 'first'
        })
        
        entities1 = set(nodes1.index)
        entities2 = set(nodes2.index)
        
        # New and lost entities
        new_entities = entities2 - entities1
        lost_entities = entities1 - entities2
        common_entities = entities1 & entities2
        
        # Compare frequencies for common entities
        comparison = []
        for entity in common_entities:
            freq1 = int(nodes1.loc[entity, 'frequency'])
            freq2 = int(nodes2.loc[entity, 'frequency'])
            change = freq2 - freq1
            pct_change = (change / freq1 * 100) if freq1 > 0 else 0
            
            comparison.append({
                'entity': entity,
                'type': nodes2.loc[entity, 'type'],
                f'{period1}_freq': freq1,
                f'{period2}_freq': freq2,
                'change': change,
                'pct_change': pct_change
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('pct_change', ascending=False)
        
        # Categorize changes
        growing = comparison_df[comparison_df['pct_change'] > 20].head(top_n)
        declining = comparison_df[comparison_df['pct_change'] < -20].head(top_n)
        stable = comparison_df[
            (comparison_df['pct_change'] >= -20) & 
            (comparison_df['pct_change'] <= 20)
        ].head(top_n)
        
        # Get details for new/lost entities
        new_df = nodes2.loc[list(new_entities)].reset_index() if new_entities else pd.DataFrame()
        lost_df = nodes1.loc[list(lost_entities)].reset_index() if lost_entities else pd.DataFrame()
        
        return {
            'new_entities': new_df,
            'lost_entities': lost_df,
            'growing': growing,
            'declining': declining,
            'stable': stable
        }
    
    def export_timeline_report(self, output_path: str) -> None:
        """
        Export a comprehensive timeline report as markdown.
        
        Args:
            output_path: Path to save the markdown report
        """
        if self.entity_timeline is None:
            self.build_entity_timeline()
        
        report_lines = [
            "# Temporal Knowledge Graph Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Dataset Overview",
            f"- **Time Range**: {self.periods[0]} to {self.periods[-1]}",
            f"- **Total Periods**: {len(self.periods)}",
            f"- **Unique Entities**: {len(self.entity_timeline)}",
            "",
            "## Entity Persistence",
            ""
        ]
        
        # Persistent entities (appear in most periods)
        persistent = self.entity_timeline[self.entity_timeline['persistence'] > 0.7].head(10)
        if not persistent.empty:
            report_lines.append("### Most Persistent Entities (>70% of periods)")
            report_lines.append("| Entity | Type | Persistence | Total Mentions |")
            report_lines.append("|--------|------|-------------|----------------|")
            for _, row in persistent.iterrows():
                report_lines.append(
                    f"| {row['entity']} | {row['type']} | "
                    f"{row['persistence']:.1%} | {row['total_mentions']} |"
                )
            report_lines.append("")
        
        # Emerging entities (first seen in later periods)
        n_periods = len(self.periods)
        emerging_threshold = self.periods[int(n_periods * 0.6)]  # Last 40% of timeline
        emerging = self.entity_timeline[
            self.entity_timeline['first_seen'] >= emerging_threshold
        ].head(10)
        if not emerging.empty:
            report_lines.append("### Emerging Entities (appeared in later periods)")
            report_lines.append("| Entity | Type | First Seen | Peak Frequency |")
            report_lines.append("|--------|------|------------|----------------|")
            for _, row in emerging.iterrows():
                report_lines.append(
                    f"| {row['entity']} | {row['type']} | "
                    f"{row['first_seen']} | {row['peak_frequency']} |"
                )
            report_lines.append("")
        
        # Top entities by total mentions
        report_lines.append("## Top 20 Entities by Total Mentions")
        report_lines.append("| Rank | Entity | Type | Total Mentions | Lifespan | Peak Period |")
        report_lines.append("|------|--------|------|----------------|----------|-------------|")
        for idx, row in self.entity_timeline.head(20).iterrows():
            report_lines.append(
                f"| {idx + 1} | {row['entity']} | {row['type']} | "
                f"{row['total_mentions']} | {row['lifespan']} | {row['peak_period']} |"
            )
        report_lines.append("")
        
        # Event detection for top entities
        report_lines.append("## Significant Events (Top 10 Entities)")
        for _, row in self.entity_timeline.head(10).iterrows():
            entity = row['entity']
            events = self.detect_events(entity, z_threshold=2.0)
            if events:
                report_lines.append(f"### {entity}")
                for event in events:
                    report_lines.append(
                        f"- **{event['period']}**: frequency={event['frequency']}, "
                        f"z-score={event['z_score']:.2f} (baseline={event['baseline']:.1f})"
                    )
                report_lines.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Timeline report saved to {output_path}")
