# Network Contagion Module

## Overview

The `contagion` module provides a comprehensive toolkit for simulating and analyzing contagion processes on networks. It supports:

- **Simple contagion models**: SI, SIS, SIR with optional thresholds
- **Complex contagion models**: Watts threshold model, k-reinforcement
- **Multiple backends**: CPU (vectorized), multiprocessing, GPU (optional)
- **Parameter inference**: Fit model parameters from observed cascades
- **Analysis utilities**: Cascade metrics, R(t), adoption curves

## Quick Start

### Simple Contagion (SI Model)

```python
import numpy as np
import pandas as pd
from src.contagion.backends import to_csr_adjacency
from src.contagion.models import SIModel
from src.contagion.simulator import run_simulation

# Create a small network
edges = pd.DataFrame({'source': [0,1,2], 'target': [1,2,0]})
n = 3
edges_array = edges[['source','target']].to_numpy()
adj = to_csr_adjacency(n, edges_array, directed=False)

# Initialize model
model = SIModel(n, np.random.default_rng(42), beta=0.5)

# Initial state: node 0 infected
init = np.zeros(n, dtype=int)
init[0] = 1

# Run simulation
result = run_simulation(model, adj, init, timesteps=10, seed=42)

# Analyze
from src.contagion.analysis import compute_cascade_metrics
metrics = compute_cascade_metrics(result.states, n)
print(f"Final size: {metrics.final_size}, Adoption rate: {metrics.adoption_rate}")
```

### Complex Contagion (Watts Model)

```python
from src.contagion.models import WattsThresholdModel

model = WattsThresholdModel(n, np.random.default_rng(42), phi=0.18)
result = run_simulation(model, adj, init, timesteps=10, seed=42)
```

### Multiprocessing Backend

```python
from src.contagion.mp_backend import MPAdjacency, run_simulation_mp

mp_adj = MPAdjacency(adj.mat)
result = run_simulation_mp(model, mp_adj, init, timesteps=10, seed=42, workers=4)
```

### Parameter Inference

```python
from src.contagion.inference import infer_parameters, ParameterSpace

edges_df = pd.DataFrame({'source': [0,1,2,3], 'target': [1,2,3,0]})
observed = {'final_size': 3, 'initial_seeds': 1}
param_space = ParameterSpace(beta_range=(0.01, 0.5), n_samples=20)

result = infer_parameters(
    model_name='si',
    edges_df=edges_df,
    observed_cascade=observed,
    param_space=param_space,
    timesteps=20,
    seed=42
)
print(f"Best params: {result.best_params}")
```

## Models

### Simple Contagion

- **SI (Susceptible-Infected)**: Once infected, stays infected
  - State codes: 0=S, 1=I
  - Parameters: `beta` (transmission rate), optional `threshold`

- **SIS (Susceptible-Infected-Susceptible)**: Can recover and become susceptible again
  - State codes: 0=S, 1=I
  - Parameters: `beta`, `gamma` (recovery rate), optional `threshold`

- **SIR (Susceptible-Infected-Recovered)**: Permanent immunity after recovery
  - State codes: 0=S, 1=I, 2=R
  - Parameters: `beta`, `gamma`, optional `threshold`

### Complex Contagion

- **Watts Threshold Model**: Adoption requires fraction `phi` of neighbors
  - State codes: 0=not adopted, 1=adopted (monotone)
  - Parameter: `phi` (threshold fraction, e.g., 0.18)

- **K-Reinforcement Model**: Adoption requires at least `k` neighbors
  - State codes: 0=not adopted, 1=adopted (monotone)
  - Parameter: `k` (integer count)

## Thresholds in Simple Contagion

Thresholds control when a susceptible node can be infected:

- `threshold < 1.0`: Fraction of neighbors (e.g., 0.25 = 25% of neighbors must be infected)
- `threshold >= 1.0`: Integer count (e.g., 2 = at least 2 infected neighbors)
- `threshold = None`: Any infected neighbor can transmit (default)

## Backends

### CPU (Default)

- Vectorized operations using NumPy and SciPy sparse matrices
- Fastest for small to medium networks (< 100k nodes)

### Multiprocessing

```python
from src.contagion.mp_backend import run_simulation_mp

result = run_simulation_mp(model, adj, init, timesteps=50, workers=8)
```

- Partitions nodes across workers
- Best for large networks on multi-core machines
- Each worker gets independent RNG stream for reproducibility

### GPU (Optional)

```python
from src.contagion.gpu_backend import to_gpu_adjacency

gpu_adj = to_gpu_adjacency(cpu_adj.mat)
result = run_simulation(model, gpu_adj, init, timesteps=50)
```

- Requires `cupy` installation
- Best for very large networks (> 1M nodes) with CUDA GPU
- Tests skip gracefully if cupy unavailable

## Analysis Utilities

```python
from src.contagion.analysis import (
    compute_cascade_metrics,
    compute_rt,
    adoption_curve,
    susceptible_infected_recovered_curves,
    events_to_dataframe
)

# Cascade summary
metrics = compute_cascade_metrics(result.states, n)
print(f"Peak time: {metrics.peak_time}, Final size: {metrics.final_size}")

# Effective reproduction number
rt = compute_rt(result.states)

# Adoption over time
df = adoption_curve(result.states)

# SIR compartment counts
sir_df = susceptible_infected_recovered_curves(result.states)

# Events log
events_df = events_to_dataframe(result.events)
```

## Command-Line Interface

### Simple Contagion

```bash
# SI simulation
python -m src.contagion.cli edges.csv --model si --beta 0.1 --timesteps 50

# SIR with custom columns
python -m src.contagion.cli edges.csv --model sir --beta 0.1 --gamma 0.05 \
  --source-col src --target-col dst --timesteps 100
```

### Complex Contagion

```bash
# Watts threshold model
python -m src.contagion.cli_complex edges.csv --model watts --phi 0.18 \
  --timesteps 50 --output-dir results/

# K-reinforcement
python -m src.contagion.cli_complex edges.csv --model k --k 2 \
  --timesteps 50 --output-dir results/
```

### Parameter Inference

```bash
# Infer SI parameters
python -m src.contagion.cli_inference edges.csv --model si \
  --observed-final-size 50 --observed-initial-seeds 2 \
  --beta-min 0.01 --beta-max 0.5 --n-samples 20 \
  --output-dir results/

# Infer SIR parameters
python -m src.contagion.cli_inference edges.csv --model sir \
  --observed-final-size 30 --observed-initial-seeds 1 \
  --beta-min 0.01 --beta-max 0.5 \
  --gamma-min 0.01 --gamma-max 0.3 \
  --n-samples 15 --search-mode random \
  --output-dir results/
```

## Output Formats

CLIs support `--output-dir` to save structured outputs:

- `events.csv`: Timestep, event type, node ID for each adoption/infection/recovery
- `summary.json`: Cascade metrics (final size, peak time, adoption rate, etc.)
- `trials.csv` (inference): All parameter combinations and scores
- `best_params.json` (inference): Best-fit parameters and score

## Best Practices

1. **Reproducibility**: Always set `seed` parameter for deterministic results
2. **Early stopping**: Use `--early-stop N` to halt when no changes for N steps
3. **Initial conditions**: Control seeding via `--patient-zero`, `--initial-frac`, or custom init state
4. **Large graphs**: Use multiprocessing backend for > 50k nodes
5. **Parameter search**: Start with coarse grid, refine around best params
6. **Thresholds**: Test both fraction-based and count-based for different network structures

## Integration with Existing Network Tools

The contagion module works seamlessly with networks built by the semantic and transformer modules:

```python
# From actor network
actor_edges = pd.read_csv('actor_edges.csv')
# Map string IDs to integers
from src.contagion.cli import _coerce_edges_to_int
edges, n = _coerce_edges_to_int(actor_edges, 'src', 'dst')
adj = to_csr_adjacency(n, edges, directed=True)

# Run contagion
model = SIModel(n, np.random.default_rng(0), beta=0.1)
init = np.zeros(n, dtype=int)
init[0] = 1
result = run_simulation(model, adj, init, timesteps=100)
```

## Testing

Run all contagion tests:

```bash
pytest tests/test_contagion*.py tests/test_mp_backend.py tests/test_analysis.py \
  tests/test_inference.py tests/test_gpu_backend.py -v
```

GPU tests skip automatically if cupy is unavailable.

## References

- Watts, D. J. (2002). "A simple model of global cascades on random networks"
- Pastor-Satorras, R., & Vespignani, A. (2001). "Epidemic spreading in scale-free networks"
- Centola, D. (2010). "The spread of behavior in an online social network experiment"

## Performance Tips

- **Small graphs (< 1k nodes)**: CPU backend is fastest
- **Medium graphs (1k-100k)**: CPU or multiprocessing
- **Large graphs (> 100k)**: Multiprocessing (4-16 workers) or GPU
- **Parameter inference**: Use `search_mode='random'` with fewer samples for faster exploration
- **Memory**: GPU backend copies graph to device; ensure sufficient VRAM

## Troubleshooting

**Q: Simulation reaches steady state too quickly**
- Try higher `beta` or lower threshold
- Check if graph has sufficient connectivity

**Q: Multiprocessing slower than CPU**
- Overhead dominates for small graphs; use CPU instead
- Try fewer workers (2-4) for medium graphs

**Q: GPU tests fail**
- Install cupy: `pip install cupy-cuda12x` (Linux/Windows) or skip (tests will be skipped)

**Q: Inference returns poor fit**
- Increase `n_samples` for finer grid
- Expand parameter ranges
- Check if model assumptions match observed cascade
