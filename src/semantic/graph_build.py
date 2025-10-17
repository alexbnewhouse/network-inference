def sparsify_topk(ppmi: Dict[Tuple[int, int], float], topk: int) -> Dict[Tuple[int, int], float]:

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def sparsify_topk_sparse(ppmi_sparse, topk: int):
    # ppmi_sparse: scipy.sparse.coo_matrix
    if topk <= 0:
        return ppmi_sparse
    from scipy import sparse
    n = ppmi_sparse.shape[0]
    # For each row, keep only topk largest
    rows, cols, data = [], [], []
    mat = ppmi_sparse.tocsr()
    for i in range(n):
        row = mat.getrow(i)
        if row.nnz == 0:
            continue
        idx = np.argsort(row.data)[-topk:]
        for j, v in zip(row.indices[idx], row.data[idx]):
            rows.append(i)
            cols.append(j)
            data.append(v)
    return sparse.coo_matrix((data, (rows, cols)), shape=ppmi_sparse.shape)



def to_networkx_sparse(ppmi_sparse, id2tok: List[str]):
    import networkx as nx
    G = nx.Graph()
    for i, tok in enumerate(id2tok):
        G.add_node(i, token=tok)
    coo = ppmi_sparse.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i != j and w > 0:
            G.add_edge(i, j, weight=float(w))
    return G

def to_igraph_sparse(ppmi_sparse, id2tok: List[str]):
    import igraph as ig
    coo = ppmi_sparse.tocoo()
    edges = [(int(i), int(j)) for i, j, w in zip(coo.row, coo.col, coo.data) if i != j and w > 0]
    weights = [float(w) for i, j, w in zip(coo.row, coo.col, coo.data) if i != j and w > 0]
    g = ig.Graph(n=len(id2tok), edges=edges)
    g.vs["token"] = id2tok
    g.es["weight"] = weights
    return g
