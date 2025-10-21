import unittest
import numpy as np

from src.contagion.backends import to_csr_adjacency
from src.contagion.models import WattsThresholdModel, KReinforcementModel
from src.contagion.simulator import run_simulation


def small_line_graph(n=5):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return np.array(edges, dtype=int)


def complete_graph(n=4):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return np.array(edges, dtype=int)


class TestComplexContagion(unittest.TestCase):
    def test_watts_threshold_adoption_on_star(self):
        # Star: center node 0 connected to 3 peripheral nodes 1,2,3
        # Center has degree 3; periphery degree 1
        # phi=0.5 means center needs 1.5 neighbors (>= 2), periphery needs 0.5 (>=1 neighbor)
        edges = np.array([(0, 1), (0, 2), (0, 3)], dtype=int)
        adj = to_csr_adjacency(4, edges, directed=False)
        model = WattsThresholdModel(4, np.random.default_rng(0), phi=0.5)
        init = np.zeros(4, dtype=int)
        init[1] = 1  # seed one peripheral node
        res = run_simulation(model, adj, init, timesteps=5, seed=1)
        # Peripheral nodes 2,3 won't adopt (only 0 neighbor adopted)
        # Center sees 1/3 < 0.5, so center won't adopt
        # No further adoption expected
        adopted_counts = [int((s == 1).sum()) for s in res.states]
        self.assertEqual(adopted_counts[0], 1)
        self.assertEqual(adopted_counts[-1], 1)

    def test_watts_threshold_cascades_on_line(self):
        # Line graph: degree 1 or 2 (endpoints vs middle)
        # phi=0.4 → endpoints need 0.4 (>=1), middle need 0.8 (>=1)
        # Seed node 0, expect 1 to adopt next step, then 2, etc.
        n = 5
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = WattsThresholdModel(n, np.random.default_rng(0), phi=0.4)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=n, seed=2)
        adopted_counts = [int((s == 1).sum()) for s in res.states]
        # Wave propagates 1 hop/step
        expected = [min(n, 1 + t) for t in range(len(res.states))]
        self.assertEqual(adopted_counts, expected)

    def test_k_reinforcement_needs_multiple(self):
        # Line graph: node 1 has neighbors {0,2}
        # k=2 → need 2 neighbors adopted
        # Seed nodes 0 and 2; node 1 should adopt next step (neighbors 0,2 both adopted)
        # Node 3 has neighbors {2,4}; only 2 is adopted, so 3 won't adopt (needs 2)
        n = 5
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = KReinforcementModel(n, np.random.default_rng(0), k=2)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        init[2] = 1
        res = run_simulation(model, adj, init, timesteps=3, seed=3)
        adopted = [int((s == 1).sum()) for s in res.states]
        # T0: 2 adopted (nodes 0,2), T1: node 1 adopts (neighbors 0,2)
        # T2+: no further adoption (nodes 3,4 don't have 2 adopted neighbors)
        self.assertEqual(adopted[0], 2)
        self.assertEqual(adopted[1], 3)
        self.assertEqual(adopted[-1], 3)

    def test_k_reinforcement_complete_graph_cascade(self):
        # Complete graph n=4: each node has 3 neighbors
        # k=2 → once 2 nodes adopt, all others adopt next step
        n = 4
        edges = complete_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = KReinforcementModel(n, np.random.default_rng(0), k=2)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        init[1] = 1
        res = run_simulation(model, adj, init, timesteps=2, seed=4)
        adopted = [int((s == 1).sum()) for s in res.states]
        self.assertEqual(adopted[0], 2)
        self.assertEqual(adopted[-1], 4)


if __name__ == "__main__":
    unittest.main()
