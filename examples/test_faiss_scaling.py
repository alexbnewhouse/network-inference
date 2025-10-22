#!/usr/bin/env python3
"""
Test the scalable transformer document network on a sample of large dataset.
"""
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.semantic.transformers_enhanced import TransformerSemanticNetwork

def test_faiss_scaling():
    """Test FAISS-based similarity search on simulated large dataset."""
    print("=" * 70)
    print("Testing FAISS-based Transformer Document Network")
    print("=" * 70)
    
    # Simulate documents
    n_docs = 15000  # Large enough to trigger FAISS
    print(f"\n📊 Simulating {n_docs:,} documents...")
    
    # Create varied synthetic documents
    topics = [
        "climate change and global warming effects",
        "artificial intelligence and machine learning",
        "cryptocurrency blockchain technology bitcoin",
        "space exploration mars colonization",
        "quantum computing quantum mechanics",
        "renewable energy solar wind power",
        "political polarization social media",
        "pandemic response public health",
        "economic inequality wealth distribution",
        "education reform online learning"
    ]
    
    documents = []
    for i in range(n_docs):
        topic = topics[i % len(topics)]
        doc = f"Document {i}: Discussion about {topic}. "
        doc += f"This is document number {i} exploring various aspects of {topic}."
        documents.append(doc)
    
    print(f"✓ Created {len(documents):,} synthetic documents")
    
    # Initialize builder
    print("\n🤖 Initializing TransformerSemanticNetwork...")
    
    # Auto-detect device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"   Using device: {device}")
    
    builder = TransformerSemanticNetwork(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )
    print("✓ Model loaded")
    
    # Test with FAISS
    print(f"\n🔍 Building document network with FAISS (top-k=20)...")
    edges_df = builder.build_document_network(
        documents,
        similarity_threshold=0.3,
        top_k=20,
        use_faiss=True
    )
    
    print(f"\n✅ Network built successfully!")
    print(f"   Documents: {len(documents):,}")
    print(f"   Edges: {len(edges_df):,}")
    print(f"   Avg edges per doc: {len(edges_df) / len(documents):.1f}")
    
    # Show sample edges
    print(f"\n📋 Sample edges:")
    print(edges_df.head(10).to_string(index=False))
    
    # Verify topic clustering
    print(f"\n🔬 Checking topic clustering...")
    # Documents 0, 10, 20, ... should be about same topic (climate)
    climate_docs = list(range(0, min(100, n_docs), 10))
    climate_edges = edges_df[
        (edges_df['source'].isin(climate_docs)) &
        (edges_df['target'].isin(climate_docs))
    ]
    print(f"   Climate topic docs: {len(climate_docs)}")
    print(f"   Within-topic edges: {len(climate_edges)}")
    print(f"   Clustering detected: {'✓' if len(climate_edges) > 50 else '✗'}")
    
    # Memory estimate for 1M docs
    print(f"\n💡 Scaling estimate for 1M documents:")
    print(f"   Memory (FAISS): ~15 GB RAM")
    print(f"   Time (CUDA): ~50 minutes")
    print(f"   Edges (top-k=20): ~10-20M edges")
    
    return edges_df

if __name__ == "__main__":
    try:
        edges_df = test_faiss_scaling()
        print("\n" + "=" * 70)
        print("✅ Test completed successfully!")
        print("=" * 70)
        print("\nYou can now process your 1M /pol/ posts with:")
        print("python -m src.semantic.transformers_cli \\")
        print("  --input pol_archive_0.csv \\")
        print("  --outdir output/pol_network \\")
        print("  --text-col body \\")
        print("  --device cuda \\")
        print("  --top-k 20")
    except ImportError as e:
        if "faiss" in str(e):
            print("\n❌ FAISS not installed!")
            print("Install with: pip install faiss-gpu (or faiss-cpu)")
        else:
            raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
