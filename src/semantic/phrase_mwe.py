"""
Phrase/Bigram/MWE Promotion
- Detect high-PMI bigrams/trigrams
- Promote as nodes in semantic network
"""
from collections import Counter, defaultdict
import itertools

class PhraseDetector:
    def __init__(self, min_count=10, min_pmi=5.0):
        self.min_count = min_count
        self.min_pmi = min_pmi

    def fit(self, docs):
        # docs: list of list of tokens
        unigram_counts = Counter()
        bigram_counts = Counter()
        total = 0
        for doc in docs:
            unigram_counts.update(doc)
            bigram_counts.update(zip(doc, doc[1:]))
            total += len(doc)
        # Compute PMI for bigrams
        pmi = {}
        for (w1, w2), c in bigram_counts.items():
            if c < self.min_count:
                continue
            p_w1 = unigram_counts[w1] / total
            p_w2 = unigram_counts[w2] / total
            p_w1w2 = c / total
            val = max(0, (p_w1w2 / (p_w1 * p_w2)))
            import math
            score = math.log2(val) if val > 0 else 0
            if score >= self.min_pmi:
                pmi[(w1, w2)] = score
        return pmi

    def promote_phrases(self, docs, pmi):
        # Replace bigrams in docs with joined phrase if in pmi
        new_docs = []
        for doc in docs:
            i = 0
            new_doc = []
            while i < len(doc):
                if i < len(doc) - 1 and (doc[i], doc[i+1]) in pmi:
                    new_doc.append(f"{doc[i]}_{doc[i+1]}")
                    i += 2
                else:
                    new_doc.append(doc[i])
                    i += 1
            new_docs.append(new_doc)
        return new_docs
