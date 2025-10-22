"""
Enhanced Sentiment Analysis for Knowledge Graphs

This module extends the basic VADER sentiment analysis with:
- Stance detection (pro/anti toward specific entities)
- Temporal sentiment trends
- Entity framing analysis (how entities are described)
- Sentiment comparison across groups/communities

Author: AI Assistant
Date: 2025-10-21
"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from pathlib import Path
import re


class StanceDetector:
    """
    Detect stance (pro/anti/neutral) toward entities based on context and sentiment.
    
    Uses pattern matching combined with sentiment to determine if a text
    expresses support or opposition toward an entity.
    """
    
    # Patterns indicating opposition/criticism
    ANTI_PATTERNS = [
        r'\b(against|oppose|anti|hate|destroy|kill|eliminate|stop|fight|reject)\b',
        r'\b(bad|terrible|awful|horrible|evil|corrupt|dangerous|threat)\b',
        r'\b(blame|fault|responsible for|caused by)\b',
        r'\b(must (go|leave|resign|be stopped))\b',
        r'\b(fuck|shit|damn|screw)\b.*\b{entity}\b',
    ]
    
    # Patterns indicating support/endorsement
    PRO_PATTERNS = [
        r'\b(support|endorse|pro|love|defend|protect|help|assist|back)\b',
        r'\b(good|great|excellent|wonderful|amazing|best|hero|leader)\b',
        r'\b(agree with|believe in|stand with|side with)\b',
        r'\b(deserve|earned|rightful)\b',
        r'\bpraise\b.*\b{entity}\b',
    ]
    
    def __init__(self):
        """Initialize stance detector."""
        self.anti_patterns = [re.compile(p, re.IGNORECASE) for p in self.ANTI_PATTERNS]
        self.pro_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRO_PATTERNS]
    
    def detect_stance(
        self, 
        text: str, 
        entity: str,
        sentiment_score: float,
        window_size: int = 100
    ) -> str:
        """
        Detect stance toward an entity in text.
        
        Args:
            text: Full text
            entity: Entity to detect stance toward
            sentiment_score: Overall sentiment score (-1 to 1)
            window_size: Character window around entity mention
        
        Returns:
            Stance: "pro", "anti", or "neutral"
        """
        text_lower = text.lower()
        entity_lower = entity.lower()
        
        # Find entity mentions
        matches = [m.start() for m in re.finditer(re.escape(entity_lower), text_lower)]
        if not matches:
            return "neutral"
        
        # Analyze context around each mention
        anti_score = 0
        pro_score = 0
        
        for match_pos in matches:
            # Extract window around mention
            start = max(0, match_pos - window_size)
            end = min(len(text), match_pos + len(entity) + window_size)
            window = text[start:end]
            
            # Check for anti patterns
            for pattern in self.anti_patterns:
                if pattern.search(window):
                    anti_score += 1
            
            # Check for pro patterns
            for pattern in self.pro_patterns:
                if pattern.search(window):
                    pro_score += 1
        
        # Combine pattern matching with sentiment
        if sentiment_score < -0.3:
            anti_score += 1
        elif sentiment_score > 0.3:
            pro_score += 1
        
        # Determine final stance
        if anti_score > pro_score:
            return "anti"
        elif pro_score > anti_score:
            return "pro"
        else:
            return "neutral"


class EntityFramingAnalyzer:
    """
    Analyze how entities are framed/described in discourse.
    
    Extracts adjectives, verbs, and descriptive phrases associated with entities
    to understand framing strategies.
    """
    
    def __init__(self):
        """Initialize framing analyzer."""
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Framing analysis will be limited.")
            self.nlp = None
    
    def extract_entity_descriptors(
        self, 
        text: str, 
        entity: str
    ) -> Dict[str, List[str]]:
        """
        Extract descriptive words and phrases associated with an entity.
        
        Args:
            text: Text containing entity
            entity: Entity to analyze
        
        Returns:
            Dictionary with keys:
                - adjectives: Adjectives modifying the entity
                - verbs: Verbs where entity is subject/object
                - compounds: Compound phrases containing entity
        """
        if self.nlp is None:
            return {'adjectives': [], 'verbs': [], 'compounds': []}
        
        doc = self.nlp(text)
        
        adjectives = []
        verbs = []
        compounds = []
        
        # Find entity mentions in doc
        entity_lower = entity.lower()
        for token in doc:
            if entity_lower in token.text.lower():
                # Get adjectives modifying this token
                for child in token.children:
                    if child.dep_ == "amod":  # Adjectival modifier
                        adjectives.append(child.text)
                    elif child.dep_ == "compound":  # Compound
                        compounds.append(f"{child.text} {token.text}")
                
                # Get verbs where entity is subject/object
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    verbs.append(token.head.text)
                elif token.dep_ in ["dobj", "pobj"]:
                    verbs.append(f"[obj] {token.head.text}")
        
        return {
            'adjectives': adjectives,
            'verbs': verbs,
            'compounds': compounds
        }
    
    def aggregate_framing(
        self, 
        texts: List[str], 
        entity: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Aggregate framing across multiple texts.
        
        Args:
            texts: List of texts
            entity: Entity to analyze
            top_n: Number of top descriptors to return
        
        Returns:
            DataFrame with descriptor frequencies
        """
        all_adjectives = []
        all_verbs = []
        all_compounds = []
        
        for text in texts:
            if entity.lower() in text.lower():
                descriptors = self.extract_entity_descriptors(text, entity)
                all_adjectives.extend(descriptors['adjectives'])
                all_verbs.extend(descriptors['verbs'])
                all_compounds.extend(descriptors['compounds'])
        
        # Count frequencies
        adj_counts = pd.Series(all_adjectives).value_counts().head(top_n)
        verb_counts = pd.Series(all_verbs).value_counts().head(top_n)
        compound_counts = pd.Series(all_compounds).value_counts().head(top_n)
        
        return pd.DataFrame({
            'adjectives': [adj_counts.to_dict() if not adj_counts.empty else {}],
            'verbs': [verb_counts.to_dict() if not verb_counts.empty else {}],
            'compounds': [compound_counts.to_dict() if not compound_counts.empty else {}]
        })


class TemporalSentimentAnalyzer:
    """
    Analyze sentiment trends over time for entities.
    
    Integrates with TemporalKG to track how sentiment toward entities
    changes across time periods.
    """
    
    def __init__(self, temporal_kg_dir: str):
        """
        Initialize temporal sentiment analyzer.
        
        Args:
            temporal_kg_dir: Directory containing time-period subdirectories
                            with sentiment-enhanced KG outputs
        """
        self.base_dir = Path(temporal_kg_dir)
        self.periods: List[str] = []
        self.sentiment_data: Dict[str, pd.DataFrame] = {}
        self._load_sentiment_data()
    
    def _load_sentiment_data(self) -> None:
        """Load sentiment data from all time periods."""
        # Find all subdirectories with sentiment data
        for root, dirs, files in sorted(self.base_dir.walk()):
            if "entity_sentiment.csv" in files:
                period = str(root.relative_to(self.base_dir))
                self.periods.append(period)
                self.sentiment_data[period] = pd.read_csv(root / "entity_sentiment.csv")
        
        if not self.periods:
            raise ValueError(f"No sentiment data found in {self.base_dir}")
        
        self.periods = sorted(self.periods)
        print(f"Loaded sentiment data from {len(self.periods)} periods")
    
    def get_entity_sentiment_trend(
        self, 
        entity: str,
        aggregation: str = "mean"
    ) -> pd.DataFrame:
        """
        Get sentiment trend for an entity across time.
        
        Args:
            entity: Entity name
            aggregation: How to aggregate sentiment per period ("mean", "median", "std")
        
        Returns:
            DataFrame with columns: period, sentiment, n_mentions
        """
        trend = []
        
        for period in self.periods:
            df = self.sentiment_data[period]
            entity_data = df[df['entity'] == entity]
            
            if not entity_data.empty:
                if aggregation == "mean":
                    agg_sentiment = entity_data['sentiment'].mean()
                elif aggregation == "median":
                    agg_sentiment = entity_data['sentiment'].median()
                elif aggregation == "std":
                    agg_sentiment = entity_data['sentiment'].std()
                else:
                    agg_sentiment = entity_data['sentiment'].mean()
                
                trend.append({
                    'period': period,
                    'sentiment': agg_sentiment,
                    'n_mentions': len(entity_data)
                })
            else:
                trend.append({
                    'period': period,
                    'sentiment': None,
                    'n_mentions': 0
                })
        
        return pd.DataFrame(trend)
    
    def detect_sentiment_shifts(
        self,
        entity: str,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect significant sentiment shifts for an entity.
        
        Args:
            entity: Entity name
            threshold: Minimum sentiment change to count as shift
        
        Returns:
            List of shift events with period, old_sentiment, new_sentiment, change
        """
        trend = self.get_entity_sentiment_trend(entity)
        trend = trend[trend['sentiment'].notna()]  # Remove periods with no data
        
        if len(trend) < 2:
            return []
        
        shifts = []
        for i in range(1, len(trend)):
            old_sent = trend.iloc[i-1]['sentiment']
            new_sent = trend.iloc[i]['sentiment']
            change = new_sent - old_sent
            
            if abs(change) >= threshold:
                shifts.append({
                    'period': trend.iloc[i]['period'],
                    'old_sentiment': old_sent,
                    'new_sentiment': new_sent,
                    'change': change,
                    'direction': 'positive' if change > 0 else 'negative'
                })
        
        return shifts
    
    def compare_entity_sentiment(
        self,
        entities: List[str],
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare sentiment across multiple entities.
        
        Args:
            entities: List of entity names
            period: Specific period to compare (None for all periods)
        
        Returns:
            DataFrame comparing sentiment metrics
        """
        if period and period not in self.periods:
            raise ValueError(f"Period {period} not found")
        
        comparison = []
        
        for entity in entities:
            if period:
                # Compare in specific period
                df = self.sentiment_data[period]
                entity_data = df[df['entity'] == entity]
                
                if not entity_data.empty:
                    comparison.append({
                        'entity': entity,
                        'period': period,
                        'mean_sentiment': entity_data['sentiment'].mean(),
                        'median_sentiment': entity_data['sentiment'].median(),
                        'std_sentiment': entity_data['sentiment'].std(),
                        'n_mentions': len(entity_data)
                    })
            else:
                # Compare across all periods
                trend = self.get_entity_sentiment_trend(entity)
                valid_trend = trend[trend['sentiment'].notna()]
                
                if not valid_trend.empty:
                    comparison.append({
                        'entity': entity,
                        'overall_mean': valid_trend['sentiment'].mean(),
                        'overall_median': valid_trend['sentiment'].median(),
                        'overall_std': valid_trend['sentiment'].std(),
                        'total_mentions': valid_trend['n_mentions'].sum(),
                        'n_periods': len(valid_trend)
                    })
        
        return pd.DataFrame(comparison)
    
    def export_sentiment_timeline(
        self,
        entities: List[str],
        output_path: str
    ) -> None:
        """
        Export sentiment timeline visualization data.
        
        Args:
            entities: List of entities to include
            output_path: Path to save CSV
        """
        all_trends = []
        
        for entity in entities:
            trend = self.get_entity_sentiment_trend(entity)
            trend['entity'] = entity
            all_trends.append(trend)
        
        combined = pd.concat(all_trends, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"Sentiment timeline exported to {output_path}")


def analyze_entity_stance_distribution(
    df: pd.DataFrame,
    entity: str,
    text_col: str = 'text'
) -> Dict[str, int]:
    """
    Analyze stance distribution toward an entity in a dataset.
    
    Args:
        df: DataFrame with text data
        entity: Entity to analyze
        text_col: Name of text column
    
    Returns:
        Dictionary with counts: {'pro': X, 'anti': Y, 'neutral': Z}
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    detector = StanceDetector()
    vader = SentimentIntensityAnalyzer()
    
    stances = {'pro': 0, 'anti': 0, 'neutral': 0}
    
    for text in df[text_col]:
        if pd.isna(text) or entity.lower() not in str(text).lower():
            continue
        
        # Get sentiment
        sentiment = vader.polarity_scores(str(text))['compound']
        
        # Detect stance
        stance = detector.detect_stance(str(text), entity, sentiment)
        stances[stance] += 1
    
    return stances


def compare_group_sentiment(
    df: pd.DataFrame,
    entity: str,
    group_col: str,
    text_col: str = 'text'
) -> pd.DataFrame:
    """
    Compare sentiment toward an entity across different groups.
    
    Args:
        df: DataFrame with text and group data
        entity: Entity to analyze
        group_col: Column identifying groups (e.g., 'board', 'user_id')
        text_col: Name of text column
    
    Returns:
        DataFrame comparing sentiment by group
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    vader = SentimentIntensityAnalyzer()
    
    # Filter to texts mentioning entity
    df_filtered = df[df[text_col].str.contains(entity, case=False, na=False)].copy()
    
    # Calculate sentiment for each text
    df_filtered['sentiment'] = df_filtered[text_col].apply(
        lambda x: vader.polarity_scores(str(x))['compound']
    )
    
    # Group by specified column
    comparison = df_filtered.groupby(group_col).agg({
        'sentiment': ['mean', 'median', 'std', 'count']
    }).round(3)
    
    comparison.columns = ['mean_sentiment', 'median_sentiment', 'std_sentiment', 'n_mentions']
    comparison = comparison.sort_values('mean_sentiment', ascending=False)
    
    return comparison.reset_index()
