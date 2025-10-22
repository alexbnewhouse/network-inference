#!/usr/bin/env python3
"""
Generate realistic training dataset for network analysis onboarding.

Creates ~10,000 social media posts with:
- Realistic text content (political, tech, cultural topics)
- User IDs (for actor network analysis)
- Timestamps (for temporal analysis)
- Thread/post structure (for conversation networks)
- Geographic and entity mentions (for knowledge graphs)
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import hashlib

# Set seed for reproducibility
random.seed(42)

# Configuration
NUM_USERS = 250
NUM_POSTS = 10000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 31)  # 3 months of data

# User personas with different posting patterns
USER_TYPES = {
    'casual': {'posts_range': (5, 20), 'weight': 0.4},
    'active': {'posts_range': (20, 60), 'weight': 0.35},
    'power': {'posts_range': (60, 150), 'weight': 0.20},
    'influencer': {'posts_range': (150, 300), 'weight': 0.05}
}

# Topic templates with realistic content
TOPICS = {
    'geopolitics': [
        "Russia and China are strengthening their alliance in {region}",
        "NATO expansion continues to be debated in {country}",
        "Ukraine conflict affects global {resource} supply chains",
        "The UN Security Council discusses {issue} in emergency session",
        "Israel and Palestine negotiations stall over {topic}",
        "Iran nuclear program raises concerns in {region}",
        "North Korea tests new {weapon} amid international pressure",
        "EU sanctions target {country} over {issue}",
        "Taiwan tensions escalate as {actor} makes statement",
        "Syria conflict enters new phase with {actor} involvement"
    ],
    'tech': [
        "AI advancement in {field} raises ethical questions",
        "New {company} product disrupts {industry} market",
        "Cybersecurity breach at {company} affects millions",
        "Cryptocurrency {coin} crashes after {event}",
        "Tech workers strike at {company} over {issue}",
        "Data privacy concerns grow around {technology}",
        "Social media platform {company} changes {policy}",
        "Quantum computing breakthrough at {institution}",
        "5G rollout continues despite {controversy}",
        "Electric vehicle sales surge in {region}"
    ],
    'politics': [
        "{politician} announces campaign for {position}",
        "Immigration policy changes spark debate in {country}",
        "Supreme Court ruling on {issue} divides nation",
        "Election fraud claims investigated in {state}",
        "Healthcare reform proposal faces opposition from {group}",
        "Climate policy debate intensifies in {region}",
        "Gun control legislation proposed after {event}",
        "Tax reform plan targets {group}",
        "Police reform protests continue in {city}",
        "Voting rights legislation stalls in {legislature}"
    ],
    'culture': [
        "New {movie} breaks box office records",
        "{celebrity} controversy sparks online debate",
        "Cancel culture discussion focuses on {person}",
        "Sports team {team} wins championship",
        "Music festival {event} sells out in minutes",
        "Art exhibition on {topic} opens at {museum}",
        "Book bans debated in {state} schools",
        "Streaming platform {company} releases {show}",
        "Fashion week highlights {trend} styles",
        "Video game {title} launches with {feature}"
    ],
    'economics': [
        "Stock market volatility continues amid {crisis}",
        "Inflation reaches {number}% in {country}",
        "Federal Reserve considers {action} on interest rates",
        "Housing market cools in {region}",
        "Unemployment drops to {number}% in {sector}",
        "Trade war between {country1} and {country2} escalates",
        "Oil prices surge after {event}",
        "Corporate merger between {company1} and {company2} announced",
        "Banking crisis affects {region} economy",
        "Minimum wage debate continues in {state}"
    ]
}

# Entity pools for realistic substitution
ENTITIES = {
    'region': ['Eastern Europe', 'Middle East', 'Asia Pacific', 'Latin America', 'Africa', 'Southeast Asia'],
    'country': ['Poland', 'Germany', 'France', 'Japan', 'India', 'Brazil', 'Turkey', 'Egypt', 'South Korea'],
    'resource': ['energy', 'grain', 'semiconductor', 'oil', 'natural gas'],
    'issue': ['human rights violations', 'territorial disputes', 'trade agreements', 'climate commitments'],
    'topic': ['settlements', 'borders', 'security guarantees', 'prisoner exchanges'],
    'actor': ['US', 'Russia', 'China', 'Turkey', 'Saudi Arabia'],
    'weapon': ['missile', 'drone', 'satellite'],
    'field': ['healthcare', 'autonomous vehicles', 'facial recognition', 'content moderation'],
    'company': ['Google', 'Meta', 'Apple', 'Microsoft', 'Tesla', 'Amazon', 'OpenAI'],
    'industry': ['transportation', 'finance', 'healthcare', 'entertainment', 'retail'],
    'event': ['regulatory news', 'hacking scandal', 'market manipulation'],
    'coin': ['Bitcoin', 'Ethereum', 'Dogecoin'],
    'technology': ['facial recognition', 'location tracking', 'targeted advertising'],
    'policy': ['algorithm transparency', 'content moderation', 'data collection'],
    'institution': ['MIT', 'Stanford', 'IBM', 'Google'],
    'controversy': ['health concerns', 'surveillance fears', 'cost issues'],
    'politician': ['Biden', 'Trump', 'DeSantis', 'Harris', 'Newsom'],
    'position': ['president', 'senate', 'governor'],
    'state': ['Texas', 'California', 'Florida', 'New York'],
    'group': ['corporations', 'middle class', 'seniors', 'small businesses'],
    'city': ['Portland', 'Minneapolis', 'Chicago', 'New York'],
    'legislature': ['Senate', 'House', 'Congress'],
    'movie': ['superhero film', 'documentary', 'action movie'],
    'celebrity': ['Kanye West', 'Taylor Swift', 'Elon Musk', 'Joe Rogan'],
    'person': ['comedian', 'athlete', 'politician', 'influencer'],
    'team': ['Lakers', 'Yankees', 'Patriots', 'Cowboys'],
    'museum': ['MoMA', 'Guggenheim', 'Met'],
    'show': ['new series', 'documentary', 'reality show'],
    'trend': ['sustainable', 'minimalist', 'vintage'],
    'title': ['FPS shooter', 'RPG', 'battle royale'],
    'feature': ['VR support', 'cross-platform play', 'AI companions'],
    'crisis': ['banking crisis', 'energy shortage', 'supply chain disruption'],
    'number': ['7.5', '4.2', '3.8', '5.1'],
    'action': ['raising', 'lowering', 'holding'],
    'sector': ['technology', 'manufacturing', 'services', 'retail'],
    'country1': ['US', 'China', 'EU'],
    'country2': ['China', 'Russia', 'Mexico'],
    'company1': ['Microsoft', 'Google', 'Amazon'],
    'company2': ['Activision', 'Twitter', 'Whole Foods']
}

# Opinion modifiers for more natural text
MODIFIERS = [
    "This is concerning because",
    "Finally some good news -",
    "Nobody is talking about how",
    "The mainstream media ignores that",
    "Breaking:",
    "Thread:",
    "Reminder:",
    "Hot take:",
    "Unpopular opinion:",
    "Change my mind:",
    "Thoughts?",
    "This aged well...",
    "Calling it now:",
]

REACTIONS = [
    "Absolutely ridiculous",
    "This is huge",
    "Wow just wow",
    "As expected",
    "Not surprised",
    "About time",
    "This won't end well",
    "Mark my words",
    "History will remember this",
    "The implications are staggering"
]

def generate_post_text(topic_category):
    """Generate realistic post text from templates."""
    template = random.choice(TOPICS[topic_category])
    
    # Fill in template with random entities
    text = template
    for key, options in ENTITIES.items():
        if '{' + key + '}' in text:
            text = text.replace('{' + key + '}', random.choice(options))
    
    # Add modifiers sometimes
    if random.random() < 0.3:
        text = f"{random.choice(MODIFIERS)} {text}"
    
    # Add reactions sometimes
    if random.random() < 0.2:
        text = f"{text}. {random.choice(REACTIONS)}."
    
    return text

def assign_users_to_posts(num_users, num_posts):
    """Assign posts to users based on user type distribution."""
    users = []
    
    for user_type, config in USER_TYPES.items():
        type_count = int(num_users * config['weight'])
        for i in range(type_count):
            user_id = f"user_{len(users):04d}"
            posts_count = random.randint(*config['posts_range'])
            users.append({
                'user_id': user_id,
                'type': user_type,
                'posts': posts_count
            })
    
    # Distribute posts
    post_assignments = []
    for user in users:
        post_assignments.extend([user['user_id']] * user['posts'])
    
    # Shuffle and trim/extend to exact count
    random.shuffle(post_assignments)
    if len(post_assignments) < num_posts:
        # Add more from random users
        while len(post_assignments) < num_posts:
            post_assignments.append(random.choice([u['user_id'] for u in users]))
    else:
        post_assignments = post_assignments[:num_posts]
    
    return post_assignments, users

def generate_threads(num_posts):
    """Generate thread structure (some posts are replies)."""
    threads = []
    thread_id = 0
    post_id = 0
    
    while post_id < num_posts:
        # Decide thread size (most threads are small)
        thread_size = random.choices(
            [1, 2, 3, 4, 5, 10, 20],
            weights=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]
        )[0]
        
        thread_size = min(thread_size, num_posts - post_id)
        
        for i in range(thread_size):
            threads.append({
                'post_id': f"post_{post_id:06d}",
                'thread_id': f"thread_{thread_id:05d}",
                'is_reply': i > 0
            })
            post_id += 1
        
        thread_id += 1
    
    return threads

def generate_timestamps(num_posts, start_date, end_date):
    """Generate realistic timestamps with temporal patterns."""
    total_seconds = int((end_date - start_date).total_seconds())
    timestamps = []
    
    for _ in range(num_posts):
        # Add some temporal clustering (more posts at certain times)
        if random.random() < 0.2:  # 20% are in "hot" periods
            # Cluster around a random date
            cluster_center = random.randint(0, total_seconds)
            offset = random.randint(-86400 * 2, 86400 * 2)  # ±2 days
            timestamp_seconds = max(0, min(total_seconds, cluster_center + offset))
        else:
            timestamp_seconds = random.randint(0, total_seconds)
        
        timestamp = start_date + timedelta(seconds=timestamp_seconds)
        timestamps.append(timestamp)
    
    timestamps.sort()  # Posts appear in chronological order
    return timestamps

def main():
    """Generate training dataset."""
    print("Generating training dataset...")
    print(f"Users: {NUM_USERS}, Posts: {NUM_POSTS}")
    
    # Generate user assignments
    user_assignments, users = assign_users_to_posts(NUM_USERS, NUM_POSTS)
    print(f"✓ Generated {len(users)} users with posting patterns")
    
    # Generate thread structure
    threads = generate_threads(NUM_POSTS)
    print(f"✓ Generated {len(set(t['thread_id'] for t in threads))} threads")
    
    # Generate timestamps
    timestamps = generate_timestamps(NUM_POSTS, START_DATE, END_DATE)
    print(f"✓ Generated timestamps from {START_DATE.date()} to {END_DATE.date()}")
    
    # Generate posts
    posts = []
    for i in range(NUM_POSTS):
        # Choose topic category
        topic = random.choice(list(TOPICS.keys()))
        
        # Generate text
        text = generate_post_text(topic)
        
        posts.append({
            'post_id': threads[i]['post_id'],
            'thread_id': threads[i]['thread_id'],
            'user_id': user_assignments[i],
            'created_at': timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'text': text,
            'is_reply': threads[i]['is_reply'],
            'topic_category': topic
        })
    
    # Create DataFrame
    df = pd.DataFrame(posts)
    
    # Add hashed user IDs for anonymized analysis
    df['user_id_hashed'] = df['user_id'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
    )
    
    # Save full dataset
    output_file = 'examples/training_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} posts to {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total posts: {len(df):,}")
    print(f"  Unique users: {df['user_id'].nunique()}")
    print(f"  Threads: {df['thread_id'].nunique()}")
    print(f"  Replies: {df['is_reply'].sum()} ({df['is_reply'].mean()*100:.1f}%)")
    print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"  Topics: {', '.join(df['topic_category'].unique())}")
    print(f"  File size: {len(df) * df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")
    
    # Save user metadata for reference
    user_df = pd.DataFrame(users)
    user_df.to_csv('examples/training_data_users.csv', index=False)
    print(f"\n✓ Saved user metadata to examples/training_data_users.csv")
    
    # Print sample
    print("\nSample posts:")
    print(df[['user_id', 'created_at', 'text']].head(3).to_string(index=False))

if __name__ == '__main__':
    main()
