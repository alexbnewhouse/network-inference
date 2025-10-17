import pandas as pd
from src.semantic.kg_pipeline import KnowledgeGraphPipeline
from src.semantic.actor_network import ActorNetworkPipeline

def test_kg_pipeline():
    df = pd.DataFrame({
        "text": ["Barack Obama met Angela Merkel.", "Elon Musk founded SpaceX."],
        "thread_id": [1, 2],
        "post_id": [1, 2],
        "author_tripcode": ["!abc", "!def"],
        "author_capcode": [None, None],
        "author_poster_id_thread": ["A", "B"]
    })
    kg = KnowledgeGraphPipeline()
    kg.run(df, ".")
    assert True

def test_actor_network():
    df = pd.DataFrame({
        "thread_id": [1, 1, 1],
        "post_id": [1, 2, 3],
        "author_tripcode": ["!abc", "!def", "!ghi"],
        "author_capcode": [None, None, None],
        "author_poster_id_thread": ["A", "B", "C"],
        "text": [">>2", ">>1 >>3", ""]
    })
    actor = ActorNetworkPipeline()
    actor.run(df, ".")
    assert True
