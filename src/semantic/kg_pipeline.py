"""
Knowledge Graph Extraction Pipeline
- Enhanced NER with entity normalization and filtering
- Entity co-occurrence for relationship inference
- Dependency parsing for relation extraction
- Property graph output (nodes/edges, GraphML/CSV)
"""
import os
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


class KnowledgeGraphPipeline:
    def __init__(self, ner_model="en_core_web_sm", min_entity_freq=2, 
                 cooccurrence_window: int | None = 100, use_dependencies=True):
        """Initialize KG extraction pipeline with improved NER.
        
        Args:
            ner_model: spaCy model name (recommend en_core_web_md or en_core_web_lg for better quality)
            min_entity_freq: Minimum frequency for entity to be included in graph
            cooccurrence_window: Character window for entity co-occurrence (None = sentence-level)
            use_dependencies: Use dependency parsing for relation extraction
        """
        import spacy  # type: ignore
        self.nlp = spacy.load(ner_model)
        self.min_entity_freq = min_entity_freq
        self.cooccurrence_window = cooccurrence_window
        self.use_dependencies = use_dependencies
        
        # Entity type filtering - keep high-value types
        self.valid_entity_types = {
            'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
            'WORK_OF_ART', 'LAW', 'LANGUAGE', 'NORP', 'FAC'
        }
        
        # Stopwords for entity cleaning
        self.entity_stopwords = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'mr', 'mrs', 'ms', 'dr', 'prof'
        }

    def _normalize_entity(self, text, label):
        """Clean and normalize entity text."""
        # Strip whitespace and lowercase for normalization
        text = text.strip()
        
        # Remove leading/trailing articles and stopwords
        words = text.split()
        while words and words[0].lower() in self.entity_stopwords:
            words = words[1:]
        while words and words[-1].lower() in self.entity_stopwords:
            words = words[:-1]
        
        if not words:
            return None
            
        normalized = ' '.join(words)
        
        # Filter out low-quality entities
        if len(normalized) < 2:
            return None
        if normalized.lower() in self.entity_stopwords:
            return None
        if re.match(r'^[\d\s\-.,]+$', normalized):  # Only numbers/punctuation
            return None
        if len(words) > 8:  # Likely extraction error
            return None
        
        # Filter common false positives
        if normalized.lower() in {'ai', 'us', 'uk', 'eu', 'ceo', 'cto', 'cfo', 'nan', 'na', 'n/a'}:
            return None
            
        # Consistent casing: Title case for names, places, and nationalities
        if label in ('PERSON', 'GPE', 'LOC', 'NORP'):
            # Title case but preserve all-caps acronyms (like USA, CIA)
            if not normalized.isupper() or len(normalized) > 4:
                normalized = normalized.title()
        elif label == 'ORG':
            # For orgs, capitalize first letter but preserve rest (e.g., "eBay", "iPhone")
            if normalized and not normalized[0].isupper():
                normalized = normalized[0].upper() + normalized[1:]
        
        return normalized

    def extract_ner(self, texts, show_progress=True):
        """Run NER with enhanced filtering and normalization."""
        # Filter out null/empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if pd.notna(text) and str(text).strip():
                valid_texts.append(str(text))
                valid_indices.append(i)
            else:
                valid_texts.append("")  # Empty placeholder
                
        docs = list(tqdm(
            self.nlp.pipe(valid_texts, batch_size=64),
            total=len(valid_texts),
            desc="Extracting entities",
            disable=not show_progress
        ))
        
        all_ents = []
        entity_counter = Counter()
        entity_contexts = defaultdict(list)  # Store contexts for each entity
        entity_doc_indices = defaultdict(set)  # Track which documents mention each entity
        
        for doc_idx, doc in enumerate(docs):
            doc_ents = []
            for ent in doc.ents:
                if ent.label_ not in self.valid_entity_types:
                    continue
                    
                normalized = self._normalize_entity(ent.text, ent.label_)
                if normalized:
                    entity_key = (normalized, ent.label_)
                    doc_ents.append({
                        'text': normalized,
                        'original': ent.text,
                        'label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char,
                        'sent': ent.sent.text if ent.sent else ""
                    })
                    entity_counter[entity_key] += 1
                    entity_doc_indices[entity_key].add(doc_idx)
                    
                    # Store first context for each entity
                    if len(entity_contexts[entity_key]) == 0:
                        context = ent.sent.text if ent.sent else doc.text[:200]
                        entity_contexts[entity_key].append(context)
            
            all_ents.append(doc_ents)
        
        # Filter entities by minimum frequency
        valid_entities = {k for k, v in entity_counter.items() if v >= self.min_entity_freq}
        
        # Filter document entities
        filtered_ents = []
        for doc_ents in all_ents:
            filtered = [e for e in doc_ents if (e['text'], e['label']) in valid_entities]
            filtered_ents.append(filtered)
        
        print(f"Extracted {len(valid_entities)} unique entities (min_freq={self.min_entity_freq})")
        
        # Return entity metadata along with counter
        entity_metadata = {
            'contexts': entity_contexts,
            'doc_indices': entity_doc_indices
        }
        return filtered_ents, entity_counter, entity_metadata

    def extract_cooccurrences(self, texts, ents_per_doc):
        """Extract entity co-occurrences within window/sentence."""
        cooccurrences = Counter()
        
        for text, doc_ents in zip(texts, ents_per_doc):
            if len(doc_ents) < 2:
                continue
            
            # Group entities by sentence or window
            if self.cooccurrence_window is None:
                # Sentence-level co-occurrence
                sent_groups = defaultdict(list)
                for ent in doc_ents:
                    sent_groups[ent['sent']].append(ent)
                
                for sent, ents in sent_groups.items():
                    for i, e1 in enumerate(ents):
                        for e2 in ents[i+1:]:
                            # Skip if same entity (prevents self-loops)
                            if e1['text'] == e2['text'] and e1['label'] == e2['label']:
                                continue
                            pair = tuple(sorted([
                                (e1['text'], e1['label']),
                                (e2['text'], e2['label'])
                            ]))
                            cooccurrences[pair] += 1
            else:
                # Character window co-occurrence
                for i, e1 in enumerate(doc_ents):
                    for e2 in doc_ents[i+1:]:
                        # Skip if same entity (prevents self-loops)
                        if e1['text'] == e2['text'] and e1['label'] == e2['label']:
                            continue
                        distance = abs(e1['start_char'] - e2['start_char'])
                        if distance <= self.cooccurrence_window:
                            pair = tuple(sorted([
                                (e1['text'], e1['label']),
                                (e2['text'], e2['label'])
                            ]))
                            cooccurrences[pair] += 1
        
        return cooccurrences

    def extract_relations_from_dependencies(self, texts, ents_per_doc):
        """Extract relations using dependency parsing."""
        if not self.use_dependencies:
            return []
        
        relations = []
        relation_counter = Counter()
        docs = list(self.nlp.pipe(texts, batch_size=64))
        
        for doc, doc_ents in zip(docs, ents_per_doc):
            if len(doc_ents) < 2:
                continue
            
            # Map entity spans to entity info (use token indices for better matching)
            ent_tokens = defaultdict(dict)
            for ent in doc_ents:
                # Find tokens that overlap with entity
                for token in doc:
                    if token.idx >= ent['start_char'] and token.idx < ent['end_char']:
                        ent_tokens[token.i] = ent
            
            # Find verbs and prepositions connecting entities
            for token in doc:
                if token.pos_ in ('VERB', 'AUX'):
                    # Find subject and object entities
                    subj_ent = None
                    obj_ent = None
                    subj_token = None
                    obj_token = None
                    
                    # Look for subjects
                    for child in token.children:
                        if child.dep_ in ('nsubj', 'nsubjpass', 'agent'):
                            # Check if child or its children are entities
                            if child.i in ent_tokens:
                                subj_ent = ent_tokens[child.i]
                                subj_token = child
                            else:
                                # Check children of compound nouns
                                for subchild in child.subtree:
                                    if subchild.i in ent_tokens:
                                        subj_ent = ent_tokens[subchild.i]
                                        subj_token = subchild
                                        break
                        
                        # Look for objects
                        if child.dep_ in ('dobj', 'attr', 'oprd'):
                            if child.i in ent_tokens:
                                obj_ent = ent_tokens[child.i]
                                obj_token = child
                            else:
                                for subchild in child.subtree:
                                    if subchild.i in ent_tokens:
                                        obj_ent = ent_tokens[subchild.i]
                                        obj_token = subchild
                                        break
                        
                        # Check prepositional objects (only if we don't have a direct object)
                        if not obj_ent and child.dep_ == 'prep':
                            for pobj in child.children:
                                if pobj.dep_ == 'pobj':
                                    if pobj.i in ent_tokens:
                                        obj_ent = ent_tokens[pobj.i]
                                        obj_token = pobj
                                    else:
                                        for subchild in pobj.subtree:
                                            if subchild.i in ent_tokens:
                                                obj_ent = ent_tokens[subchild.i]
                                                obj_token = subchild
                                                break
                    
                    # Add relation if we found both subject and object
                    if subj_ent and obj_ent and subj_ent['text'] != obj_ent['text']:
                        # Create a cleaner predicate
                        predicate = token.lemma_
                        if token.pos_ == 'AUX':
                            # For auxiliary verbs, try to get the main verb
                            for child in token.children:
                                if child.pos_ == 'VERB':
                                    predicate = child.lemma_
                                    break
                        
                        rel_key = (subj_ent['text'], predicate, obj_ent['text'])
                        relation_counter[rel_key] += 1
        
        # Convert to list of relations with counts
        for (subj, pred, obj), count in relation_counter.items():
            # Get types from first occurrence
            relations.append({
                'subject': subj,
                'predicate': pred,
                'object': obj,
                'count': count
            })
        
        return relations

    def build_property_graph(self, ents_per_doc, entity_counter, cooccurrences, relations, entity_metadata=None):
        """Build property graph as nodes/edges DataFrames."""
        # Build nodes with statistics
        nodes = []
        ent2id = {}
        node_id = 0
        
        for (text, label), count in entity_counter.items():
            if count >= self.min_entity_freq:
                ent2id[(text, label)] = node_id
                node_dict = {
                    "id": node_id,
                    "entity": text,
                    "type": label,
                    "frequency": count
                }
                
                # Add metadata if available
                if entity_metadata:
                    entity_key = (text, label)
                    if entity_key in entity_metadata.get('contexts', {}):
                        contexts = entity_metadata['contexts'][entity_key]
                        node_dict['first_context'] = contexts[0][:200] if contexts else ""
                    if entity_key in entity_metadata.get('doc_indices', {}):
                        node_dict['n_unique_contexts'] = len(entity_metadata['doc_indices'][entity_key])
                
                nodes.append(node_dict)
                node_id += 1
        
        # Build edges from co-occurrences
        edges = []
        for (e1, e2), count in cooccurrences.items():
            if e1 in ent2id and e2 in ent2id:
                edges.append({
                    "source": ent2id[e1],
                    "target": ent2id[e2],
                    "source_entity": e1[0],
                    "target_entity": e2[0],
                    "weight": count,
                    "relation_type": "co-occurrence"
                })
        
        # Add relation edges from dependency parsing
        for rel in relations:
            # Find matching entity keys (need to match by text across all types)
            subj_id = None
            obj_id = None
            
            for (ent_text, ent_type), eid in ent2id.items():
                if ent_text == rel['subject']:
                    subj_id = eid
                if ent_text == rel['object']:
                    obj_id = eid
            
            if subj_id is not None and obj_id is not None:
                edges.append({
                    "source": subj_id,
                    "target": obj_id,
                    "source_entity": rel['subject'],
                    "target_entity": rel['object'],
                    "weight": rel['count'],
                    "relation_type": rel['predicate']
                })
        
        return pd.DataFrame(nodes), pd.DataFrame(edges)

    def run(self, df, outdir, show_progress=True):
        """Run full KG extraction pipeline."""
        texts = df["text"].astype(str).tolist()
        
        # Extract and filter entities
        ents_per_doc, entity_counter, entity_metadata = self.extract_ner(texts, show_progress=show_progress)
        
        # Extract co-occurrences
        print("Extracting entity co-occurrences...")
        cooccurrences = self.extract_cooccurrences(texts, ents_per_doc)
        print(f"Found {len(cooccurrences)} entity co-occurrence pairs")
        
        # Extract relations from dependencies
        relations = []
        if self.use_dependencies:
            print("Extracting relations from dependencies...")
            relations = self.extract_relations_from_dependencies(texts, ents_per_doc)
            print(f"Found {len(relations)} dependency-based relations")
        
        # Build graph
        nodes_df, edges_df = self.build_property_graph(
            ents_per_doc, entity_counter, cooccurrences, relations, entity_metadata
        )
        
        # Save outputs
        os.makedirs(outdir, exist_ok=True)
        nodes_df.to_csv(os.path.join(outdir, "kg_nodes.csv"), index=False)
        edges_df.to_csv(os.path.join(outdir, "kg_edges.csv"), index=False)
        
        print(f"\nKnowledge Graph Summary:")
        print(f"  Nodes (entities): {len(nodes_df)}")
        print(f"  Edges (relationships): {len(edges_df)}")
        if len(nodes_df) > 0:
            print(f"  Entity types: {nodes_df['type'].value_counts().to_dict()}")
        
        # Generate quality report
        if len(nodes_df) > 0 and len(edges_df) > 0:
            try:
                from .kg_quality_report import generate_kg_report
                generate_kg_report(nodes_df, edges_df, outdir, df)
            except Exception as e:
                print(f"Note: Could not generate quality report: {e}")
        
        # Return entities_per_doc for potential sentiment analysis
        return nodes_df, edges_df, ents_per_doc
