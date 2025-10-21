import unittest
import numpy as np
import pandas as pd

from src.contagion.inference import infer_parameters, ParameterSpace


class TestInference(unittest.TestCase):
    def test_infer_si_parameters(self):
        # Small triangle graph
        edges = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 0]})
        # Simulate a "ground truth" cascade: final_size=3 (all infected)
        observed = {"final_size": 3, "initial_seeds": 1}
        param_space = ParameterSpace(beta_range=(0.1, 1.0), n_samples=5, search_mode="grid")
        result = infer_parameters(
            model_name="si",
            edges_df=edges,
            observed_cascade=observed,
            param_space=param_space,
            timesteps=5,
            seed=10,
        )
        self.assertIn("beta", result.best_params)
        self.assertLess(result.best_score, 1.0)  # should fit reasonably well

    def test_infer_sir_parameters(self):
        # Line graph
        edges = pd.DataFrame({"source": [0, 1, 2, 3], "target": [1, 2, 3, 4]})
        observed = {"final_size": 3, "initial_seeds": 1}
        param_space = ParameterSpace(beta_range=(0.1, 1.0), gamma_range=(0.1, 0.5), n_samples=3, search_mode="random")
        result = infer_parameters(
            model_name="sir",
            edges_df=edges,
            observed_cascade=observed,
            param_space=param_space,
            timesteps=10,
            seed=20,
        )
        self.assertIn("beta", result.best_params)
        self.assertIn("gamma", result.best_params)
        self.assertGreater(len(result.all_results), 0)


if __name__ == "__main__":
    unittest.main()
