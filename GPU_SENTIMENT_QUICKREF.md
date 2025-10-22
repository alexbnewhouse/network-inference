# GPU Sentiment Quick Reference

## Installation (One Command)

```bash
# For NVIDIA GPU (RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers accelerate
```

## Basic Usage

```bash
# VADER (fast, CPU)
python -m src.semantic.kg_cli --input data.csv --outdir output/ --add-sentiment

# Transformer (accurate, GPU)
python -m src.semantic.kg_cli --input data.csv --outdir output/ --add-sentiment --sentiment-model transformer --sentiment-device cuda
```

## Models

| Model | Command | Speed | Accuracy | Best For |
|-------|---------|-------|----------|----------|
| **VADER** | `--sentiment-model vader` | ⚡⚡⚡⚡⚡ | 75% | Quick exploration |
| **Twitter-RoBERTa** | `--sentiment-model transformer` | ⚡⚡⚡ | 90% | Social media (default) |
| **BERT Multilingual** | `--sentiment-transformer-model nlptown/bert-base-multilingual-uncased-sentiment` | ⚡⚡ | 88% | Multi-language |
| **DistilBERT** | `--sentiment-transformer-model distilbert-base-uncased-finetuned-sst-2-english` | ⚡⚡⚡⚡ | 85% | Lightweight |

## All CLI Options

```bash
python -m src.semantic.kg_cli \
    --input data.csv \
    --outdir output/ \
    --add-sentiment \
    --sentiment-model transformer \                    # vader or transformer
    --sentiment-transformer-model [MODEL_NAME] \      # HuggingFace model
    --sentiment-device cuda \                          # cpu, cuda, or mps
    --sentiment-batch-size 128                         # Higher = faster (until OOM)
```

## Performance Matrix (RTX 5090)

| Documents | VADER Time | Transformer Time | Transformer Gain |
|-----------|-----------|------------------|------------------|
| 10K | 10s | 67s | Better accuracy |
| 100K | 1.7m | 11m | 13% more correct |
| 1M | 17m | 111m | 130K more correct |

## Output Files

All methods produce:
- `kg_nodes_with_sentiment.csv` - Entities with sentiment
- `entity_sentiment.csv` - Detailed statistics
- `sentiment_summary.txt` - Human-readable report

## Decision Tree

```
Do you need high accuracy on nuanced sentiment?
├─ YES: Use transformer (--sentiment-model transformer)
│   ├─ Have GPU? Use --sentiment-device cuda
│   └─ No GPU? Use --sentiment-device cpu (slower)
│
└─ NO: Use VADER (--sentiment-model vader)
    └─ Fast enough for millions of documents
```

## Dissertation Workflow

```bash
# Step 1: Extract KG with GPU sentiment
python -m src.semantic.kg_cli \
    --input pol_data.csv \
    --outdir output/pol_kg \
    --model en_core_web_md \
    --add-sentiment \
    --sentiment-model transformer \
    --sentiment-device cuda \
    --sentiment-batch-size 128

# Step 2: Analyze temporal trends (Python)
from src.semantic.kg_sentiment_enhanced import TemporalSentimentAnalyzer
temporal = TemporalSentimentAnalyzer()
trends_df, fig = temporal.analyze_temporal_sentiment(
    df=pol_data,
    entities_of_interest=["Breivik", "Tarrant", "Roof"],
    time_col="timestamp",
    text_col="post_content",
    window="30D"
)
fig.savefig("output/canonization_timeline.png")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not detected | `nvidia-smi` to check drivers |
| Out of memory | Reduce `--sentiment-batch-size` |
| Slow on CPU | Use `--sentiment-device cuda` |
| Model download fails | `export HF_HOME=/path/to/cache` |

## Test Commands

```bash
# Test transformer installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Compare VADER vs Transformer
python examples/compare_sentiment_models.py

# Test on training data (10K posts)
python -m src.semantic.kg_cli \
    --input examples/training_data.csv \
    --outdir output/test \
    --add-sentiment \
    --sentiment-model transformer \
    --sentiment-device cuda
```

## Documentation

- **Full Guide**: [GPU_SENTIMENT_GUIDE.md](GPU_SENTIMENT_GUIDE.md)
- **Summary**: [GPU_SENTIMENT_SUMMARY.md](GPU_SENTIMENT_SUMMARY.md)
- **Examples**: `examples/compare_sentiment_models.py`
- **Help**: `python -m src.semantic.kg_cli --help`

## Model URLs

- twitter-roberta: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
- bert-multilingual: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
- distilbert: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
