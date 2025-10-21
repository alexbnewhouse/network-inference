import unittest
import numpy as np

from src.contagion.backends import to_csr_adjacency
from src.contagion.gpu_backend import CUPY_AVAILABLE, to_gpu_adjacency


def small_line_graph(n=5):
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1))
    return np.array(edges, dtype=int)


@unittest.skipIf(not CUPY_AVAILABLE, "cupy not available")
class TestGPUBackend(unittest.TestCase):
    def test_gpu_adjacency_counts(self):
        n = 5
        edges = small_line_graph(n)
        cpu_adj = to_csr_adjacency(n, edges, directed=False)
        gpu_adj = to_gpu_adjacency(cpu_adj.mat)
        active = np.array([1, 0, 1, 0, 0], dtype=int)
        cpu_counts = cpu_adj.infected_neighbor_counts(active)
        gpu_counts = gpu_adj.infected_neighbor_counts(active)
        np.testing.assert_array_equal(cpu_counts, gpu_counts)

    def test_gpu_degrees(self):
        n = 5
        edges = small_line_graph(n)
        cpu_adj = to_csr_adjacency(n, edges, directed=False)
        gpu_adj = to_gpu_adjacency(cpu_adj.mat)
        cpu_deg = cpu_adj.degrees()
        gpu_deg = gpu_adj.degrees()
        np.testing.assert_array_equal(cpu_deg, gpu_deg)


if __name__ == "__main__":
    unittest.main()
