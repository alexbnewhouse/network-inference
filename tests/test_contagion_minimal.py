import unittest
import numpy as np

from src.contagion.backends import to_csr_adjacency
from src.contagion.models import SIModel, SIRModel, SISModel
from src.contagion.simulator import run_simulation


def small_line_graph(n=5):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return np.array(edges, dtype=int)


class TestContagionMinimal(unittest.TestCase):
    def test_si_runs_and_infects_monotonic(self):
        n = 6
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        rng = np.random.default_rng(0)
        model = SIModel(n, rng, beta=1.0)  # deterministic infection
        init = np.zeros(n, dtype=int)
        init[0] = 1
        T = 4
        res = run_simulation(model, adj, init, timesteps=T, seed=123)
        infected_counts = [s.sum() for s in res.states]
        self.assertEqual(infected_counts, sorted(infected_counts))
        expected = min(n, 1 + T)  # infection wave advances 1 hop/step from endpoint
        self.assertEqual(int(infected_counts[-1]), expected)

    def test_sir_recovery(self):
        n = 4
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        rng = np.random.default_rng(0)
        model = SIRModel(n, rng, beta=1.0, gamma=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=3, seed=7)
        self.assertTrue(any((s == 2).any() for s in res.states))

    def test_sis_churns(self):
        n = 3
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        rng = np.random.default_rng(0)
        model = SISModel(n, rng, beta=1.0, gamma=0.5)
        init = np.zeros(n, dtype=int)
        init[1] = 1
        res = run_simulation(model, adj, init, timesteps=5, seed=42)
        self.assertGreaterEqual(len(res.states), 2)


if __name__ == "__main__":
    unittest.main()
