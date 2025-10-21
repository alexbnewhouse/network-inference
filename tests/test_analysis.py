import unittest
import numpy as np

from src.contagion.backends import to_csr_adjacency
from src.contagion.models import SIModel, SIRModel
from src.contagion.simulator import run_simulation
from src.contagion.analysis import (
    compute_cascade_metrics,
    compute_rt,
    adoption_curve,
    susceptible_infected_recovered_curves,
    events_to_dataframe,
)


def small_line_graph(n=5):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return np.array(edges, dtype=int)


class TestAnalysis(unittest.TestCase):
    def test_cascade_metrics(self):
        n = 5
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = SIModel(n, np.random.default_rng(0), beta=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=6, seed=1)
        metrics = compute_cascade_metrics(res.states, n)
        self.assertEqual(metrics.initial_seeds, 1)
        self.assertEqual(metrics.final_size, n)
        self.assertEqual(metrics.adoption_rate, 1.0)

    def test_rt_computation(self):
        n = 4
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = SIModel(n, np.random.default_rng(0), beta=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=4, seed=2)
        rt = compute_rt(res.states)
        # R(t) should be positive when infections grow
        self.assertTrue(all(rt >= 0))

    def test_adoption_curve(self):
        n = 3
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = SIModel(n, np.random.default_rng(0), beta=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=3, seed=3)
        df = adoption_curve(res.states)
        self.assertEqual(len(df), len(res.states))
        self.assertEqual(df["adopted"].iloc[0], 1)
        self.assertGreater(df["adopted"].iloc[-1], df["adopted"].iloc[0])

    def test_sir_curves(self):
        n = 4
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = SIRModel(n, np.random.default_rng(0), beta=1.0, gamma=0.5)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=5, seed=4)
        df = susceptible_infected_recovered_curves(res.states)
        self.assertEqual(len(df), len(res.states))
        # Initially all S except one I
        self.assertEqual(df["S"].iloc[0], n - 1)
        self.assertEqual(df["I"].iloc[0], 1)
        self.assertEqual(df["R"].iloc[0], 0)

    def test_events_to_dataframe(self):
        n = 3
        edges = small_line_graph(n)
        adj = to_csr_adjacency(n, edges, directed=False)
        model = SIModel(n, np.random.default_rng(0), beta=1.0)
        init = np.zeros(n, dtype=int)
        init[0] = 1
        res = run_simulation(model, adj, init, timesteps=2, seed=5)
        df = events_to_dataframe(res.events)
        self.assertGreater(len(df), 0)
        self.assertIn("timestep", df.columns)
        self.assertIn("event", df.columns)
        self.assertIn("node", df.columns)


if __name__ == "__main__":
    unittest.main()
