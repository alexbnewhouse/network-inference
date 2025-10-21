import unittest
import numpy as np
import pandas as pd

from src.contagion.cli import SimpleSimConfig, simulate_from_edges_df


class TestContagionCLI(unittest.TestCase):
    def test_simulate_from_edges_df(self):
        # triangle graph
        edges = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]})
        cfg = SimpleSimConfig(model="si", beta=1.0, gamma=None, timesteps=3, seed=0, early_stop=0, directed=False, patient_zero=0, initial_frac=None)
        res = simulate_from_edges_df(edges, cfg)
        # With beta=1.0 on triangle, all infected by step 2 at most.
        infected_counts = [int((s == 1).sum()) for s in res.states]
        self.assertGreaterEqual(infected_counts[-1], 2)


if __name__ == "__main__":
    unittest.main()
