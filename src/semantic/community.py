"""
Community Detection and Reporting
- Run Louvain/Leiden on semantic graphs
- Output top communities and keywords
"""
import networkx as nx
from collections import Counter

class CommunityDetector:
    def __init__(self, method="louvain"):
        self.method = method

    def detect(self, G):
        if self.method == "louvain":
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight="weight")
        elif self.method == "leiden":
            import igraph as ig
            import leidenalg
            g = ig.Graph.from_networkx(G)
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
            # Map igraph node ids to tokens
            partition = {v.index: part for part, comm in enumerate(partition) for v in comm}
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return partition

    def summarize(self, G, partition, id2tok, topn=10):
        # For each community, get top-n tokens by degree
        comm2nodes = {}
        for node, comm in partition.items():
            comm2nodes.setdefault(comm, []).append(node)
        summaries = []
        for comm, nodes in comm2nodes.items():
            degrees = Counter({n: G.degree(n) for n in nodes})
            top_nodes = [id2tok[n] for n, _ in degrees.most_common(topn)]
            summaries.append({"community": comm, "size": len(nodes), "top_tokens": top_nodes})
        return summaries
