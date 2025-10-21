"""Contagion simulation and inference package.

Modules:
- models: SI, SIS, SIR, Watts, KReinforcement
- backends: CSR adjacency (CPU)
- mp_backend: Multiprocessing support
- simulator: Deterministic run loop
- analysis: Cascade metrics and visualization helpers
- inference: Parameter fitting from observed cascades
- cli: Simple contagion CLI
- cli_complex: Complex contagion CLI
- cli_inference: Parameter inference CLI
"""

from . import (
    models,
    backends,
    mp_backend,
    simulator,
    analysis,
    inference,
    cli,
    cli_complex,
    cli_inference,
)

__all__ = [
    "models",
    "backends",
    "mp_backend",
    "simulator",
    "analysis",
    "inference",
    "cli",
    "cli_complex",
    "cli_inference",
]
