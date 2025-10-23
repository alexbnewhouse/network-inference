#!/usr/bin/env python3
"""
Generate sample 4chan-style data for testing and demonstration.

This creates realistic 4chan post data with:
- Anonymous posts (no tripcodes)
- Optional tripcodes for some users
- Reply patterns (>>12345678)
- Board-specific content
- Temporal patterns
- Thread structure
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_4chan_sample(n_posts=500, boards=None, start_date=None):
    """
    Generate sample 4chan-style data.
    
    Args:
        n_posts: Number of posts to generate
        boards: List of boards to include (default: ['pol', 'int', 'sci'])
        start_date: Starting datetime (default: 30 days ago)
    
    Returns:
        DataFrame with 4chan-style columns
    """
    if boards is None:
        boards = ['pol', 'int', 'sci']
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    
    # Sample entities and topics by board
    board_entities = {
        'pol': {
            'entities': ['Trump', 'Biden', 'Russia', 'China', 'Ukraine', 'NATO', 'EU', 'Putin', 
                        'Zelensky', 'Congress', 'FBI', 'CIA', 'Israel', 'Palestine'],
            'topics': ['election', 'immigration', 'economy', 'foreign policy', 'corruption',
                      'media bias', 'deep state', 'sanctions', 'war', 'borders'],
            'verbs': ['announced', 'criticized', 'supported', 'opposed', 'investigated',
                     'denied', 'confirmed', 'threatened', 'promised', 'failed']
        },
        'int': {
            'entities': ['France', 'Germany', 'UK', 'Japan', 'India', 'Brazil', 'Mexico',
                        'Canada', 'Australia', 'Italy', 'Spain', 'Netherlands'],
            'topics': ['culture', 'language', 'history', 'travel', 'food', 'customs',
                      'traditions', 'holidays', 'music', 'art'],
            'verbs': ['celebrates', 'practices', 'enjoys', 'prefers', 'dislikes',
                     'embraces', 'rejects', 'values', 'criticizes', 'appreciates']
        },
        'sci': {
            'entities': ['NASA', 'SpaceX', 'CERN', 'MIT', 'Stanford', 'Cambridge',
                        'Nobel Prize', 'Nature', 'Science', 'Einstein', 'Hawking'],
            'topics': ['quantum physics', 'astronomy', 'biology', 'climate change',
                      'artificial intelligence', 'medicine', 'space exploration',
                      'genetics', 'neuroscience', 'chemistry'],
            'verbs': ['discovered', 'proved', 'demonstrated', 'published', 'researched',
                     'found', 'showed', 'revealed', 'tested', 'measured']
        }
    }
    
    # Reply patterns
    reply_patterns = [
        ">>%d Nice try shill",
        ">>%d This is exactly right",
        ">>%d You're glowing",
        ">>%d Source?",
        ">>%d Checked and correct",
        ">>%d Cope",
        ">>%d Based take",
        ">>%d Cringe opinion",
    ]
    
    # Common 4chan phrases
    common_phrases = [
        "Anyone have sources on this?",
        "This is actually important",
        "Why is nobody talking about this?",
        "Happening thread",
        "We need to discuss this",
        "Check out these numbers",
        "The absolute state of",
        "You can't make this up",
        "This is huge if true",
        "Nothing ever happens",
    ]
    
    # Generate posts
    posts = []
    post_no = 90000000  # Starting post number
    thread_ids = []
    
    # Create some threads
    n_threads = n_posts // 10  # ~10 posts per thread
    for _ in range(n_threads):
        thread_ids.append(post_no)
        post_no += random.randint(1, 100)
    
    # Track tripcodes (5% of posts use tripcodes)
    tripcodes = [
        "!Ep8pui8Vw2", "!TRUMPa4VHM", "!2hu.pKvGp6", 
        "!ITPGxtuP.A", "!ZNGZuPPu.E", "!Q1N2W3E4R5"
    ]
    
    for i in range(n_posts):
        board = random.choice(boards)
        thread_id = random.choice(thread_ids)
        
        # 5% of posts have tripcodes
        tripcode = random.choice(tripcodes) if random.random() < 0.05 else None
        
        # Generate timestamp (random within 30 days)
        timestamp = start_date + timedelta(
            seconds=random.randint(0, 30 * 24 * 60 * 60)
        )
        
        # Generate post content based on board
        board_data = board_entities[board]
        
        # 30% chance of reply
        if random.random() < 0.3 and i > 0:
            # Reply to previous post
            reply_to = posts[random.randint(max(0, i-20), i-1)]['no']
            template = random.choice(reply_patterns)
            content = template % reply_to + " "
        else:
            content = ""
        
        # Add main content
        if random.random() < 0.3:
            # Common phrase
            content += random.choice(common_phrases) + " "
        
        # Entity-focused post
        entity = random.choice(board_data['entities'])
        verb = random.choice(board_data['verbs'])
        topic = random.choice(board_data['topics'])
        
        content += f"{entity} {verb} {topic}. "
        
        # Add secondary entity (40% chance)
        if random.random() < 0.4:
            entity2 = random.choice(board_data['entities'])
            if entity2 != entity:
                content += f"This relates to {entity2}. "
        
        # Add opinion (30% chance)
        opinions = [
            "This is concerning.",
            "Finally some good news.",
            "Nothing will change.",
            "This affects everyone.",
            "The implications are huge.",
            "Nobody saw this coming.",
            "Exactly what I expected.",
            "Things are getting worse.",
            "This is progress.",
            "Extremely based."
        ]
        if random.random() < 0.3:
            content += random.choice(opinions)
        
        posts.append({
            'no': post_no,
            'thread_id': thread_id,
            'board': board,
            'body': content.strip(),
            'time': int(timestamp.timestamp()),
            'created_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'tripcode': tripcode,
            'name': 'Anonymous'
        })
        
        post_no += random.randint(1, 50)
    
    return pd.DataFrame(posts)


def main():
    """Generate and save sample 4chan data."""
    print("Generating sample 4chan-style data...")
    
    # Generate 500 posts across 3 boards
    df = generate_4chan_sample(
        n_posts=500,
        boards=['pol', 'int', 'sci']
    )
    
    # Save full version
    output_file = 'examples/sample_4chan.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df)} posts to {output_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total posts: {len(df)}")
    print(f"  Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"  Boards: {df['board'].value_counts().to_dict()}")
    print(f"  Posts with tripcodes: {df['tripcode'].notna().sum()} ({df['tripcode'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  Unique threads: {df['thread_id'].nunique()}")
    print(f"  Reply posts: {df['body'].str.contains('>>', regex=False).sum()}")
    
    # Show sample
    print("\nSample posts:")
    print(df[['board', 'body', 'tripcode']].head(5).to_string(index=False))
    
    print("\n✓ Sample 4chan data generated successfully!")
    print(f"\nTry it:")
    print(f"  python -m src.semantic.kg_cli --input {output_file} --text-col body --outdir output/chan_test")


if __name__ == "__main__":
    main()
