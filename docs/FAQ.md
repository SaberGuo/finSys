# FAQ & Troubleshooting

## Data Issues

### Q: `xtquant` fails with authentication error

**Cause**: xtquant requires a licensed QMT (迅投) client running on the same machine.

**Fix**: The pipeline automatically falls back to `akshare`, then `baostock`. Set `source_priority` in `config.yaml`:

```yaml
data:
  source_priority: [akshare, baostock]
```

### Q: Data has NaN values after preprocessing

**Cause**: Suspension days (停牌) leave gaps in price data.

**Fix**: `preprocess_market_data()` automatically forward-fills `close` on suspension days and sets `volume=0`. If NaN remains, check that you have at least one valid trading day per ticker in the date range.

### Q: `ValueError: close price must be > 0`

**Cause**: Raw data contains adjusted prices that went negative or zero (reverse split artifacts).

**Fix**: Verify the date range doesn't span a delisting event. Remove the affected ticker from `config.yaml::stocks`.

---

## Training Issues

### Q: `ValueError: stock_dim=N but df contains M unique tickers`

**Cause**: The Parquet file has fewer tickers than expected (some tickers had no data in the date range).

**Fix**: Check `df['tic'].unique()` and update `config.yaml::stocks` to match.

### Q: Training crashes with `CUDA out of memory`

**Cause**: Large observation space or batch size.

**Fix**: Reduce `training.ppo.batch_size` or number of stocks. For ≤ 30 stocks the baseline model fits in 4 GB VRAM.

### Q: Backtest Sharpe is very low (< 0)

**Cause**: Too few training timesteps or random initialization.

**Fix**: Increase `training.total_timesteps` to at least 100,000. PPO typically needs 500k–2M steps for meaningful convergence on stock trading.

---

## Qwen / Sentiment Issues

### Q: `QwenLoadError: Failed to load Qwen model`

**Cause 1**: Model weights not downloaded.

**Fix**: Run `huggingface-cli download Qwen/Qwen2.5-7B-Instruct` or set `TRANSFORMERS_OFFLINE=0`.

**Cause 2**: Insufficient VRAM.

**Fix**: Enable 4-bit quantization in config (default):

```yaml
sentiment:
  quantize_4bit: true
```

Requires ≥ 8 GB VRAM. For CPU-only, set `quantize_4bit: false` (slow but functional).

### Q: Sentiment scores are all `0.0`

**Cause**: Qwen is returning invalid JSON → graceful degradation to neutral.

**Fix**: Check logs for `"Failed to parse Qwen response"` warnings. Verify `max_new_tokens ≥ 128`.

---

## Feature Fusion Issues

### Q: Enhanced dataset has same row count but fusion columns are all `0.0`

**Cause**: Sentiment JSONL tickers don't match market data tickers.

**Fix**: Ensure both use dot-notation format (e.g. `000001.SZ`, not `SZ000001`).

### Q: `KeyError: 'sentiment_score'` during enhanced training

**Cause**: Passing baseline dataset to `build_env()` with extended indicator list.

**Fix**: Use enhanced Parquet (output of `finsys fuse`) for enhanced training, not the baseline one.

---

## Testing Issues

### Q: `ModuleNotFoundError: No module named 'pandas'`

**Fix**: Install dependencies first:

```bash
pip install -r requirements.txt
```

### Q: Integration tests hang or timeout

**Cause**: Integration tests require real model training which is slow.

**Fix**: Integration tests are opt-in. Run only unit + contract tests for fast feedback:

```bash
pytest tests/unit tests/contract -q
```

For integration tests with small datasets:

```bash
pytest tests/integration -m integration -q
```

### Q: Coverage drops below 80%

**Fix**: Run `pytest --cov=finquant --cov-report=html` and check `htmlcov/index.html` for uncovered lines. The `nlp/model.py` GPU loading path and `cli/main.py` are excluded from coverage requirements.
