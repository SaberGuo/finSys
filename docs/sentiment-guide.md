# Sentiment Analysis Guide

## Overview

`finSys` uses [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with 4-bit NF4 quantization to extract bullish/bearish sentiment scores from Chinese financial news texts.

## Hardware Requirements

| Config | Requirement |
|--------|------------|
| 4-bit quantized (default) | ≥ 8 GB VRAM (GPU) |
| CPU-only fallback | ≥ 16 GB RAM, slow |

## Prepare Input Data

Create a JSONL file where each line is:

```json
{"date": "2024-01-02", "tic": "000001.SZ", "text": "平安银行发布强劲年报，营收增长15%。"}
```

Required fields: `date` (YYYY-MM-DD), `tic` (dot-notation), `text`.

## Run Sentiment Analysis

```bash
finsys sentiment analyze \
  --config config/default.yaml.example \
  --input data/news/news_2024.jsonl \
  --output data/sentiment/
```

Output: `data/sentiment/sentiment_records.jsonl`

Each output line:

```json
{"date": "2024-01-02", "tic": "000001.SZ", "score": 0.78, "label": "positive"}
```

## Sentiment Score Interpretation

| Range | Label | Meaning |
|-------|-------|---------|
| `0.3` to `1.0` | `positive` | Bullish signal |
| `-0.3` to `0.3` | `neutral` | No clear signal |
| `-1.0` to `-0.3` | `negative` | Bearish signal |

## Graceful Degradation

If the Qwen model fails to load or produces invalid output, a **neutral** score (`0.0`) is returned for that record. The pipeline continues without interruption.

Enable/disable sentiment in `config.yaml`:

```yaml
sentiment:
  enabled: true
  model_id: Qwen/Qwen2.5-7B-Instruct
  quantize_4bit: true
  max_new_tokens: 512
  batch_size: 8
```

## Fuse with Market Data

After generating sentiment records:

```bash
finsys fuse \
  --market data/processed/train_data.parquet \
  --sentiment data/sentiment/sentiment_records.jsonl \
  --output data/enhanced/enhanced_train.parquet
```

The enhanced dataset includes 7 additional columns:
`sentiment_score`, `event_count`, `has_positive_event`, `has_negative_event`,
`revenue_growth_pct`, `net_profit_margin`, `debt_ratio`.

Missing combinations are filled with neutral defaults (see [data-model.md](data-model.md) §EnhancedDataset).

## Compare Baseline vs Enhanced

```bash
finsys compare \
  --baseline-report reports/ppo_20240131_metrics.csv \
  --enhanced-report reports/ppo_enhanced_20240131_metrics.csv
```

Outputs a JSON table with delta values for each metric.
