# Semantic Network Study (4chan /pol/ sample)

This repository contains a minimal pipeline to build a semantic co-occurrence graph (PPMI-weighted) from a sample dataset `full_pol.csv`.

## Data

Expected CSV columns:
- `text`: Post/comment content (string)
- `subject`: Thread subject or topic (string, optional)

## Quick start

1) Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Build a semantic network from `full_pol.csv` and write graph files to `out/`.

```bash
python -m src.semantic.build_semantic_network --input full_pol.csv --outdir out --min-df 5 --window 10 --topk 20
```

Outputs:
- `out/nodes.parquet`: vocabulary with frequencies.
- `out/edges.parquet`: PPMI-weighted co-occurrence edges.
- `out/graph.graphml`: graph file for visualization.

## Notes
- This pipeline is tuned for an initial pass: it includes simple normalization, stopword removal, and quote/greentext stripping heuristics.
- For temporal analysis (monthly slices) and more advanced tokenization/phrase detection, see future milestones.