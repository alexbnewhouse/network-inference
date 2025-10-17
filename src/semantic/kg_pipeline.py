"""
Knowledge Graph Extraction Pipeline
- NER (spaCy or transformers)
- Entity linking (Wikidata or gazetteer)
- Relation extraction (pattern-based, dependency-based)
- Property graph output (nodes/edges, GraphML/CSV)
"""
import os
import pandas as pd
from tqdm import tqdm

class KnowledgeGraphPipeline:
    def __init__(self, ner_model="en_core_web_sm", linker=None, rel_patterns=None):
        import spacy
        # Prefer a lightweight default model for portability; allow override via CLI
        self.nlp = spacy.load(ner_model)
        self.linker = linker  # Optional: function or class for entity linking
        self.rel_patterns = rel_patterns or []  # List of (pattern, rel_type)

    def extract_ner(self, texts):
        """Run NER and return list of entities per text."""
        docs = list(self.nlp.pipe(texts, batch_size=64))
        ents = []
        for doc in docs:
            ents.append([(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents])
        return ents

    def link_entities(self, ents):
        """Stub: link entities to KB (Wikidata, gazetteer, etc)."""
        # For now, just return surface forms
        return [[(text, label, None) for (text, label, _s, _e) in ent_list] for ent_list in ents]

    def extract_relations(self, texts, ents):
        """Stub: pattern-based relation extraction."""
        # For now, just return empty
        return [[] for _ in texts]

    def build_property_graph(self, df, ents_linked, rels):
        """Build property graph as nodes/edges DataFrames."""
        nodes = []
        edges = []
        node_id = 0
        ent2id = {}
        for i, ent_list in enumerate(ents_linked):
            for text, label, kb_id in ent_list:
                key = (text, label, kb_id)
                if key not in ent2id:
                    ent2id[key] = node_id
                    nodes.append({"id": node_id, "text": text, "label": label, "kb_id": kb_id})
                    node_id += 1
                # Edge: post -> entity
                edges.append({"src": f"post_{i}", "dst": ent2id[key], "type": "mention"})
        # Add relation edges
        for i, rel_list in enumerate(rels):
            for subj, rel, obj in rel_list:
                edges.append({"src": subj, "dst": obj, "type": rel})
        return pd.DataFrame(nodes), pd.DataFrame(edges)

    def run(self, df, outdir):
        texts = df["text"].astype(str).tolist()
        ents = self.extract_ner(texts)
        ents_linked = self.link_entities(ents)
        rels = self.extract_relations(texts, ents_linked)
        nodes_df, edges_df = self.build_property_graph(df, ents_linked, rels)
        os.makedirs(outdir, exist_ok=True)
        nodes_df.to_csv(os.path.join(outdir, "kg_nodes.csv"), index=False)
        edges_df.to_csv(os.path.join(outdir, "kg_edges.csv"), index=False)
        print(f"KG nodes: {len(nodes_df)}, edges: {len(edges_df)}")
