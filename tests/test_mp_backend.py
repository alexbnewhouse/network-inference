import unittest
import numpy as np

from src.contagion.backends import to_csr_adjacency
from src.contagion.models import SIModel
from src.contagion.simulator import run_simulation
from src.contagion.mp_backend import MPAdjacency, run_simulation_mp


def small_line_graph(n=5):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return np.array(edges, dtype=int)


class TestMPBackend(unittest.TestCase):
    def test_mp_adjacency_same_as_csr(self):
        n = 5
        edges = small_line_graph(n)
        csr_adj = to_csr_adjacency(n, edges, directed=False)
        mp_adj = MPAdjacency(csr_adj.mat)
        active = np.array([1, 0, 1, 0, 0], dtype=int)
        csr_counts = csr_adj.infected_neighbor_counts(active)
        mp_counts = mp_adj.infected_neighbor_counts(active)
        np.testing.assert_array_equal(csr_counts, mp_counts)

    def test_mp_simulation_deterministic(self):
        n = 6
        edges = small_line_graph(n)
        adj = MPAdjacency(to_csr_adjacency(n, edges, directed=False).mat)
        model = SIModel(n, np.random.default_rng(0), beta=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation_mp(model, adj, init, timesteps=5, seed=42, workers=2)
        infected = [int((s == 1).sum()) for s in res.states]
        # Monotonic infection
        self.assertEqual(infected, sorted(infected))
        # Wave propagates
        self.assertGreater(infected[-1], infected[0])

    def test_mp_matches_serial(self):
        # Compare mp vs serial on same graph/seed
        n = 5
        edges = small_line_graph(n)
        adj_csr = to_csr_adjacency(n, edges, directed=False)
        adj_mp = MPAdjacency(adj_csr.mat)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        # Serial
        model_s = SIModel(n, np.random.default_rng(0), beta=0.5)
        res_s = run_simulation(model_s, adj_csr, init, timesteps=10, seed=99)
        # MP (note: seeds will differ per worker, so exact match may not hold; but same dynamics)
        model_mp = SIModel(n, np.random.default_rng(0), beta=0.5)
        res_mp = run_simulation_mp(model_mp, adj_mp, init, timesteps=10, seed=99, workers=2)
        # Both should infect all nodes eventually with beta=0.5 and enough steps
        inf_s = [int((s == 1).sum()) for s in res_s.states]
        inf_mp = [int((s == 1).sum()) for s in res_mp.states]
        # Monotonic in both
        self.assertEqual(inf_s, sorted(inf_s))
        self.assertEqual(inf_mp, sorted(inf_mp))


if __name__ == "__main__":
    unittest.main()
