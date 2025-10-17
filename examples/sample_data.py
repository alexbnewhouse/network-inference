"""
Generate sample datasets for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_news_dataset(n_docs=100, seed=42):
    """Generate a sample news headlines dataset."""
    np.random.seed(seed)
    
    topics = {
        'AI/Tech': [
            "AI model achieves breakthrough in",
            "Tech company announces new",
            "Machine learning algorithm improves",
            "Researchers develop advanced",
            "Neural network demonstrates",
            "Quantum computing makes progress in",
            "Software update introduces",
            "Cybersecurity experts warn about",
        ],
        'Climate': [
            "Climate change impacts",
            "Global temperatures reach",
            "Renewable energy production",
            "Carbon emissions decline in",
            "Scientists discover climate",
            "Green technology advances",
            "Environmental activists protest",
            "Sustainable development goals",
        ],
        'Finance': [
            "Stock market reaches",
            "Cryptocurrency prices surge",
            "Economic indicators show",
            "Federal Reserve announces",
            "Investment firm reports",
            "Financial regulation changes",
            "Banking sector experiences",
            "Market volatility increases",
        ],
        'Health': [
            "Medical breakthrough in",
            "Healthcare system faces",
            "New treatment shows promise",
            "Public health officials warn",
            "Clinical trial results reveal",
            "Hospital capacity reaches",
            "Vaccine development progresses",
            "Mental health awareness",
        ],
    }
    
    endings = [
        "according to latest research",
        "in major development",
        "experts say",
        "study finds",
        "as reported today",
        "officials confirm",
        "data shows",
        "analysis reveals",
    ]
    
    headlines = []
    categories = []
    timestamps = []
    
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_docs):
        # Pick a random topic
        topic_name = np.random.choice(list(topics.keys()))
        prefix = np.random.choice(topics[topic_name])
        ending = np.random.choice(endings)
        
        headline = f"{prefix} {ending}"
        headlines.append(headline)
        categories.append(topic_name)
        
        # Random timestamp within last year
        random_days = np.random.randint(0, 365)
        timestamp = start_date + timedelta(days=random_days)
        timestamps.append(timestamp)
    
    df = pd.DataFrame({
        'id': range(n_docs),
        'text': headlines,
        'category': categories,
        'timestamp': timestamps,
        'subject': [f"thread_{i%20}" for i in range(n_docs)],  # For actor networks
    })
    
    return df


def generate_forum_posts(n_posts=200, seed=42):
    """Generate sample forum discussion posts."""
    np.random.seed(seed)
    
    post_templates = [
        "I think {} is really important because {}",
        "Has anyone tried {}? I'm curious about {}",
        "Just wanted to share my experience with {}. {}",
        "Looking for advice on {}. Any suggestions about {}?",
        ">>>{} - I agree with this! Also, {}",
        "Great point about {}! I'd add that {}",
        "Does anyone know more about {}? Specifically {}",
    ]
    
    topics = [
        "machine learning", "climate change", "cryptocurrency",
        "renewable energy", "artificial intelligence", "data science",
        "neural networks", "blockchain technology", "solar panels",
        "electric vehicles", "quantum computing", "cybersecurity"
    ]
    
    details = [
        "it affects everyone", "the technology is mature",
        "there are many applications", "it's cost-effective now",
        "the research is promising", "it solves real problems",
        "adoption is growing", "experts recommend it"
    ]
    
    posts = []
    threads = []
    post_ids = []
    
    for i in range(n_posts):
        template = np.random.choice(post_templates)
        topic = np.random.choice(topics)
        detail = np.random.choice(details)
        
        post = template.format(topic, detail)
        posts.append(post)
        threads.append(f"thread_{i % 10}")
        post_ids.append(i)
    
    df = pd.DataFrame({
        'post_id': post_ids,
        'thread_id': threads,
        'text': posts,
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_posts)]
    })
    
    return df


def generate_research_abstracts(n_abstracts=50, seed=42):
    """Generate sample research paper abstracts."""
    np.random.seed(seed)
    
    templates = [
        "We present a novel approach to {} using {}. Our method achieves {} and demonstrates {}. Experiments on {} show that our approach outperforms existing methods by {}.",
        "This paper introduces {} for {}. We evaluate our technique on {} and find that it improves {} while maintaining {}. Our results suggest that {}.",
        "Recent advances in {} have enabled new applications in {}. We propose {} which achieves {}. Evaluation on {} confirms that our method provides {}.",
    ]
    
    methods = ["deep learning", "transformer models", "graph neural networks", 
               "reinforcement learning", "meta-learning", "self-supervised learning"]
    applications = ["natural language processing", "computer vision", "speech recognition",
                   "recommendation systems", "autonomous driving", "medical diagnosis"]
    results = ["state-of-the-art performance", "significant improvements", 
               "robust generalization", "efficient computation"]
    datasets = ["benchmark datasets", "real-world data", "large-scale corpora"]
    
    abstracts = []
    
    for i in range(n_abstracts):
        template = np.random.choice(templates)
        abstract = template.format(
            np.random.choice(applications),
            np.random.choice(methods),
            np.random.choice(results),
            np.random.choice(results),
            np.random.choice(datasets),
            f"{np.random.randint(5, 30)}%"
        )
        abstracts.append(abstract)
    
    df = pd.DataFrame({
        'id': range(n_abstracts),
        'text': abstracts,
        'title': [f"Paper {i+1}: {np.random.choice(methods).title()}" for i in range(n_abstracts)],
        'year': np.random.choice([2022, 2023, 2024, 2025], n_abstracts)
    })
    
    return df


if __name__ == "__main__":
    # Generate and save sample datasets
    print("Generating sample datasets...")
    
    # News dataset
    news_df = generate_news_dataset(n_docs=100)
    news_df.to_csv("sample_news.csv", index=False)
    print(f"✓ Created sample_news.csv ({len(news_df)} documents)")
    
    # Forum posts
    forum_df = generate_forum_posts(n_posts=200)
    forum_df.to_csv("sample_forum.csv", index=False)
    print(f"✓ Created sample_forum.csv ({len(forum_df)} posts)")
    
    # Research abstracts
    research_df = generate_research_abstracts(n_abstracts=50)
    research_df.to_csv("sample_research.csv", index=False)
    print(f"✓ Created sample_research.csv ({len(research_df)} abstracts)")
    
    print("\n✅ All sample datasets generated!")
    print("\nPreview of news dataset:")
    print(news_df.head())
