from __future__ import annotations

from pathlib import Path

import click

from finquant.config.settings import load_config
from finquant.data.pipeline import DataPipeline


@click.group()
def cli() -> None:
    """finSys CLI entrypoint."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@cli.group()
def data() -> None:
    """Data commands."""


@data.command("fetch")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option("--output", "output_dir", default="data/processed", show_default=True)
@click.option("--start", "start_date", default=None)
@click.option("--end", "end_date", default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def data_fetch(
    config_path: str,
    output_dir: str,
    start_date: str | None,
    end_date: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Download market data and save to Parquet."""
    config = load_config(config_path)
    if start_date:
        config.dates.train_start = start_date
    if end_date:
        config.dates.test_end = end_date

    if dry_run:
        click.echo("Config loaded successfully.")
        return

    if verbose:
        click.echo(f"Fetching {len(config.stocks)} symbols...")

    pipeline = DataPipeline(config)
    output = pipeline.fetch_and_save(Path(output_dir))
    click.echo(f"Saved to {output}")


@data.command("build-5min")
@click.option("--config", "config_path", default="config/csi500_5min.yaml", show_default=True)
@click.option("--indicator-set", "indicator_set_id", default=None, help="Indicator set ID to use.")
@click.option("--all-indicator-sets", is_flag=True, default=False, help="Generate datasets for all configured indicator sets.")
@click.option("--output", "output_path", default=None, help="Output Parquet file path.")
@click.option("--output-dir", "output_dir", default="data/training", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def data_build_5min(
    config_path: str,
    indicator_set_id: str | None,
    all_indicator_sets: bool,
    output_path: str | None,
    output_dir: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Build 5-minute training dataset from zz500_data.db."""
    import pandas as pd

    from finquant.config.settings import AppConfig, IndicatorSetConfig, TargetConfig
    from finquant.data.dataset import TrainingDatasetBuilder
    from finquant.data.pipeline import DataPipeline
    from finquant.features.indicator_sets import IndicatorSetRegistry

    config = load_config(config_path)

    if dry_run:
        click.echo("[dry-run] Would build 5min training dataset.")
        return

    if verbose:
        click.echo(f"Building 5min dataset from {config_path} ...")

    # Fetch 5min raw data
    pipeline = DataPipeline(config)
    raw_df = pipeline.fetch()

    if verbose:
        click.echo(f"Fetched {len(raw_df)} rows, {raw_df['tic'].nunique()} tickers")

    # Build registry from config
    registry = IndicatorSetRegistry.from_configs(config.indicator_sets)

    builder = TrainingDatasetBuilder(config, registry=registry)

    if all_indicator_sets:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for iset_id in registry.list_ids():
            out = out_dir / f"{iset_id}_{config.dates.train_start}_{config.dates.train_end}.parquet"
            builder.build(raw_df, indicator_set_id=iset_id, output_path=out)
            click.echo(f"Saved {iset_id} dataset to {out}")
    else:
        if indicator_set_id is None:
            indicator_set_id = config.indicator_sets[0].id if config.indicator_sets else None
        out = output_path or str(Path(output_dir) / f"{indicator_set_id}_dataset.parquet")
        builder.build(raw_df, indicator_set_id=indicator_set_id, output_path=out)
        click.echo(f"Saved dataset to {out}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command("train")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option(
    "--data-file",
    "data_file",
    required=True,
    help="Path to processed Parquet file (output of `data fetch`).",
)
@click.option("--output", "output_dir", default="models", show_default=True)
@click.option(
    "-a",
    "--algo",
    "algorithm",
    type=click.Choice(["ppo", "sac", "td3"], case_sensitive=False),
    default=None,
    help="Override training algorithm from config.",
)
@click.option("-t", "--timesteps", "total_timesteps", type=int, default=None)
@click.option(
    "--mode",
    "mode",
    type=click.Choice(["trading", "scoring"], case_sensitive=False),
    default=None,
    help="Training mode: trading (multi-stock portfolio) or scoring (single-stock scoring).",
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def train_cmd(
    config_path: str,
    data_file: str,
    output_dir: str,
    algorithm: str | None,
    total_timesteps: int | None,
    mode: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Train a FinRL agent on processed market data."""
    import pandas as pd

    from finquant.training.trainer import Trainer

    config = load_config(config_path)
    if algorithm:
        config.training.algorithm = algorithm
    if total_timesteps:
        config.training.total_timesteps = total_timesteps
    if mode:
        config.training.mode = mode

    training_mode = config.training.mode

    if dry_run:
        click.echo(
            f"[dry-run] Would train {config.training.algorithm} in {training_mode} mode "
            f"for {config.training.total_timesteps} steps on {data_file}"
        )
        return

    df = pd.read_parquet(data_file)
    stock_dim = df["tic"].nunique()

    # Validate mode vs data
    if training_mode == "scoring" and stock_dim != 1:
        click.echo(
            f"Error: Scoring mode requires single-stock data, but {data_file} contains {stock_dim} stocks."
        )
        click.echo("Hint: Filter data to a single stock or use --mode trading")
        return

    if verbose:
        if training_mode == "scoring":
            ticker = df["tic"].iloc[0]
            click.echo(
                f"Training {config.training.algorithm} in scoring mode on {ticker} "
                f"({df['date'].nunique()} days) ..."
            )
        else:
            click.echo(
                f"Training {config.training.algorithm} in trading mode on {stock_dim} stocks "
                f"× {df['date'].nunique()} days ..."
            )

    trainer = Trainer(config)
    model_path = trainer.train(train_df=df, output_dir=Path(output_dir), mode=training_mode)
    click.echo(f"Model saved to {model_path}")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@cli.command("backtest")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option(
    "--model",
    "model_path",
    required=True,
    help="Path to saved SB3 model (.zip).",
)
@click.option(
    "--data-file",
    "data_file",
    required=True,
    help="Path to test Parquet file.",
)
@click.option("--output", "output_dir", default="reports", show_default=True)
@click.option("--risk-free-rate", "risk_free_rate", type=float, default=0.02, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def backtest_cmd(
    config_path: str,
    model_path: str,
    data_file: str,
    output_dir: str,
    risk_free_rate: float,
    verbose: bool,
) -> None:
    """Run backtest and generate HTML + CSV reports."""
    import json

    import pandas as pd

    from finquant.training.trainer import Trainer

    config = load_config(config_path)
    df = pd.read_parquet(data_file)

    if verbose:
        click.echo(f"Backtesting model {model_path} on {df['date'].nunique()} days ...")

    trainer = Trainer(config)
    report = trainer.backtest(
        model_path=Path(model_path),
        test_df=df,
        output_dir=Path(output_dir),
        risk_free_rate=risk_free_rate,
    )
    metrics = report.to_dict()
    click.echo(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# news
# ---------------------------------------------------------------------------

@cli.group()
def news() -> None:
    """News fetching commands."""


@news.command("fetch")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option("--output", "output_dir", default="data/news", show_default=True)
@click.option("--max-pages", "max_pages", type=int, default=5, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def news_fetch(
    config_path: str,
    output_dir: str,
    max_pages: int,
    verbose: bool,
) -> None:
    """Fetch East Money news for all stocks in config."""
    from finquant.data.news_fetcher import fetch_news_for_universe

    config = load_config(config_path)
    out = fetch_news_for_universe(
        tickers=config.stocks,
        output_dir=output_dir,
        max_pages=max_pages,
        verbose=verbose,
    )
    click.echo(f"News saved to {out}")


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------

@cli.group()
def fundamentals() -> None:
    """Fundamental data commands."""


@fundamentals.command("fetch")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option("--output", "output_dir", default="data/fundamentals", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def fundamentals_fetch(
    config_path: str,
    output_dir: str,
    verbose: bool,
) -> None:
    """Fetch financial reports and compute fundamental metrics for all stocks."""
    import pandas as pd

    from finquant.data.fundamental_fetcher import fetch_fundamentals_for_universe
    from finquant.data.pipeline import DataPipeline

    config = load_config(config_path)

    # Fetch market data to extract trading dates
    pipeline = DataPipeline(config)
    market_df = pipeline.fetch()
    trading_dates = sorted(market_df["date"].unique().tolist())

    out = fetch_fundamentals_for_universe(
        tickers=config.stocks,
        trading_dates=trading_dates,
        output_dir=output_dir,
        verbose=verbose,
    )
    click.echo(f"Fundamentals saved to {out}")


# ---------------------------------------------------------------------------
# sentiment
# ---------------------------------------------------------------------------

@cli.group()
def sentiment() -> None:
    """Sentiment analysis commands."""


@sentiment.command("analyze")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option(
    "--input",
    "input_file",
    required=True,
    help="JSONL file with {date, tic, text} records.",
)
@click.option("--output", "output_dir", default="data/sentiment", show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def sentiment_analyze(
    config_path: str,
    input_file: str,
    output_dir: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run Qwen sentiment analysis on financial news texts."""
    from finquant.features.sentiment import SentimentProcessor

    config = load_config(config_path)

    if dry_run:
        click.echo(f"[dry-run] Would analyze {input_file} with Qwen.")
        return

    processor = SentimentProcessor(config.sentiment)
    out = processor.process_file(
        input_path=Path(input_file),
        output_dir=Path(output_dir),
        verbose=verbose,
    )
    click.echo(f"Sentiment records saved to {out}")


# ---------------------------------------------------------------------------
# fuse / compare
# ---------------------------------------------------------------------------

@cli.command("fuse")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option(
    "--market",
    "market_file",
    required=True,
    help="Processed market Parquet file.",
)
@click.option(
    "--sentiment",
    "sentiment_file",
    default=None,
    help="Sentiment JSONL file (output of `sentiment analyze`).",
)
@click.option(
    "--fundamentals",
    "fundamental_file",
    default=None,
    help="Fundamental JSONL file (output of `fundamentals fetch`).",
)
@click.option("--output", "output_file", default="data/enhanced/dataset.parquet", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def fuse_cmd(
    config_path: str,
    market_file: str,
    sentiment_file: str | None,
    fundamental_file: str | None,
    output_file: str,
    verbose: bool,
) -> None:
    """Fuse market data with sentiment / fundamental features."""
    import pandas as pd

    from finquant.features.fusion import fuse_datasets

    market_df = pd.read_parquet(market_file)
    out_path = fuse_datasets(
        market_df=market_df,
        sentiment_file=Path(sentiment_file) if sentiment_file else None,
        fundamental_file=Path(fundamental_file) if fundamental_file else None,
        output_path=Path(output_file),
        verbose=verbose,
    )
    click.echo(f"Enhanced dataset saved to {out_path}")


@cli.command("compare")
@click.option(
    "--baseline-report",
    "baseline_report",
    required=True,
    help="Baseline backtest CSV metrics file.",
)
@click.option(
    "--enhanced-report",
    "enhanced_report",
    required=True,
    help="Enhanced backtest CSV metrics file.",
)
def compare_cmd(baseline_report: str, enhanced_report: str) -> None:
    """Compare baseline vs enhanced backtest metrics."""
    import json

    import pandas as pd

    base = pd.read_csv(baseline_report).set_index("metric")["value"].to_dict()
    enhanced = pd.read_csv(enhanced_report).set_index("metric")["value"].to_dict()

    metrics = sorted(set(base) | set(enhanced))
    rows = []
    for m in metrics:
        b = base.get(m, float("nan"))
        e = enhanced.get(m, float("nan"))
        diff = e - b if isinstance(b, float) and isinstance(e, float) else "n/a"
        rows.append({"metric": m, "baseline": b, "enhanced": e, "delta": diff})

    click.echo(json.dumps(rows, indent=2))


@cli.command("compare-multi")
@click.option("--run-dir", "run_dir", required=True, help="Directory containing model run subdirectories.")
@click.option("--output", "output_path", default=None, help="Output CSV/JSON path.")
def compare_multi_cmd(run_dir: str, output_path: str | None) -> None:
    """Compare multiple indicator-set models and output ranked summary."""
    from finquant.training.comparison import ComparisonAnalyzer

    analyzer = ComparisonAnalyzer(run_dir)
    report = analyzer.run()
    df = report.to_dataframe()

    if output_path:
        p = Path(output_path)
        if p.suffix == ".csv":
            report.to_csv(p)
        elif p.suffix == ".json":
            report.to_json(p)
        else:
            report.to_csv(p.with_suffix(".csv"))
            report.to_json(p.with_suffix(".json"))
        click.echo(f"Report saved to {p.with_suffix('.csv')} and {p.with_suffix('.json')}")
    else:
        click.echo(df.to_string(index=False))


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

@cli.group()
def selection() -> None:
    """Stock selection commands."""


@selection.command("run")
@click.option("--config", "config_path", default="config/selection_test.yaml", show_default=True)
@click.option("--start", "start_date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", "end_date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--output-dir", "output_dir", default="data/selection", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def selection_run(
    config_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    verbose: bool,
) -> None:
    """Run stock selection for a date range and save results."""
    import pandas as pd
    import logging

    from finquant.data.pipeline import DataPipeline
    from finquant.selection import create_strategy

    # Configure logging level based on verbose flag
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )

    config = load_config(config_path)

    if verbose:
        click.echo(f"Running selection from {start_date} to {end_date}...")

    # Update config date range to match command-line arguments
    # Add buffer for historical factor calculation (e.g., 250 days for annual metrics)
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    buffer_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    config.dates.train_start = buffer_start
    config.dates.test_end = end_date

    # Fetch market data (daily)
    pipeline = DataPipeline(config)
    market_df = pipeline.fetch()

    # Fetch index data
    if config.selection is None:
        click.echo("Error: config.selection is required")
        return

    index_ticker = config.selection.index_ticker
    index_config = config.model_copy()
    index_config.stocks = [index_ticker]
    index_pipeline = DataPipeline(index_config)
    index_df = index_pipeline.fetch()

    # Compute required technical indicators for market state classification
    from finquant.features.technical import compute_indicators
    index_df = compute_indicators(index_df, indicators=["adx_14", "atr_14"])

    # Convert date to string format for selection pipeline (if not already)
    if pd.api.types.is_datetime64_any_dtype(index_df["date"]):
        index_df["date"] = index_df["date"].dt.strftime("%Y-%m-%d")
    if pd.api.types.is_datetime64_any_dtype(market_df["date"]):
        market_df["date"] = market_df["date"].dt.strftime("%Y-%m-%d")

    # Initialize selection strategy with verbose flag
    strategy = create_strategy(config, verbose=verbose)

    # Get trading days in range
    trading_days = sorted(
        [d for d in market_df["date"].unique() if start_date <= d <= end_date]
    )

    if verbose:
        click.echo(f"Processing {len(trading_days)} trading days...")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for date in trading_days:
        try:
            result = strategy.select(market_df, index_df, date)
            result.save(str(out_dir / f"{date}_selection.json"))
            results.append(result)

            if not verbose:
                # Simple output when not verbose
                click.echo(
                    f"{date}: {result.market_state.value}, "
                    f"{len(result.selected_tickers)} stocks selected"
                )
        except Exception as e:
            click.echo(f"Error on {date}: {e}")

    click.echo(f"\nCompleted: {len(results)} selection results saved to {out_dir}")


@selection.command("backtest")
@click.option("--config", "config_path", default="config/selection_test.yaml", show_default=True)
@click.option("--model", "model_path", default=None, help="Path to trained RL model (.zip), required for --mode rl")
@click.option("--mode", "mode", type=click.Choice(["rl", "simple"]), default="rl", show_default=True, help="Backtest mode: rl (with trained RL model) or simple (fixed stop-loss/take-profit)")
@click.option("--stop-loss", "stop_loss_pct", type=float, default=-0.05, show_default=True, help="Stop loss percentage (simple mode only)")
@click.option("--take-profit", "take_profit_pct", type=float, default=0.10, show_default=True, help="Take profit percentage (simple mode only)")
@click.option("--start", "start_date", required=True, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", "end_date", required=True, help="Backtest end date (YYYY-MM-DD)")
@click.option("--output-dir", "output_dir", default="runs/dynamic_backtest", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def selection_backtest(
    config_path: str,
    model_path: str | None,
    mode: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    start_date: str,
    end_date: str,
    output_dir: str,
    verbose: bool,
) -> None:
    """Run dynamic backtest: daily selection + trading."""
    import pandas as pd
    from pathlib import Path

    from finquant.data.pipeline import DataPipeline
    from finquant.selection import SelectionPipeline
    from finquant.training.dynamic_backtest import DynamicBacktester, SimpleBacktester

    # Validate model path for RL mode
    if mode == "rl":
        if model_path is None:
            click.echo("Error: --model is required when using --mode rl")
            click.echo("Usage: finsys selection backtest --model <path-to-model.zip> --mode rl ...")
            return
        # Check if model file exists
        model_file = Path(model_path)
        if not model_file.exists() and not model_file.with_suffix(".zip").exists():
            click.echo(f"Error: Model file not found: {model_path}")
            click.echo("Hint: Did you mean --mode simple instead of --model simple?")
            return
    # If user passed a non-zip value to --model while using simple mode, warn them
    if mode == "simple" and model_path is not None and not str(model_path).endswith(".zip"):
        click.echo(f"Warning: --model '{model_path}' is ignored in simple mode")
        click.echo("Hint: Use --mode rl --model <path> for RL mode")
        model_path = None

    config = load_config(config_path)

    if verbose:
        click.echo(f"Running {mode} backtest from {start_date} to {end_date}...")
        if mode == "rl":
            click.echo(f"Using model: {model_path}")
        else:
            click.echo(f"Stop loss: {stop_loss_pct:.1%}, Take profit: {take_profit_pct:.1%}")

    # Update config date range
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    buffer_start = (start_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    config.dates.train_start = buffer_start
    config.dates.test_end = end_date

    # Fetch market data (daily)
    if verbose:
        click.echo("Fetching market data...")
    pipeline = DataPipeline(config)
    market_df = pipeline.fetch()

    # Fetch index data
    if config.selection is None:
        click.echo("Error: config.selection is required")
        return

    index_ticker = config.selection.index_ticker
    index_config = config.model_copy()
    index_config.stocks = [index_ticker]
    index_pipeline = DataPipeline(index_config)
    index_df = index_pipeline.fetch()

    # Compute required technical indicators for market state classification
    from finquant.features.technical import compute_indicators
    index_df = compute_indicators(index_df, indicators=["adx_14", "atr_14"])

    # Convert date to string format
    if pd.api.types.is_datetime64_any_dtype(index_df["date"]):
        index_df["date"] = index_df["date"].dt.strftime("%Y-%m-%d")
    if pd.api.types.is_datetime64_any_dtype(market_df["date"]):
        market_df["date"] = market_df["date"].dt.strftime("%Y-%m-%d")

    # Initialize selection pipeline
    selection_pipeline = SelectionPipeline.from_config(config)

    # Initialize backtester based on mode
    if verbose:
        click.echo(f"Initializing {mode} backtester...")

    if mode == "rl":
        backtester = DynamicBacktester(
            config=config,
            model_path=Path(model_path),
            selection_pipeline=selection_pipeline,
        )
    else:
        backtester = SimpleBacktester(
            config=config,
            selection_pipeline=selection_pipeline,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    # Run backtest
    if verbose:
        click.echo("Running backtest...")
    report = backtester.run(
        market_df=market_df,
        index_df=index_df,
        start_date=start_date,
        end_date=end_date,
    )

    # Save report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"backtest_{start_date}_{end_date}.json"
    report.save(report_path)

    # Print summary
    mode_label = "RL" if mode == "rl" else "Simple"
    click.echo("\n" + "=" * 60)
    click.echo(f"{mode_label} Backtest Results")
    click.echo("=" * 60)
    click.echo(f"\nPeriod: {start_date} to {end_date}")
    click.echo(f"Trading days: {len(report.daily_records)}")
    click.echo(f"\nInitial amount: ¥{report.initial_amount:,.2f}")
    click.echo(f"Final value: ¥{report.final_value:,.2f}")
    click.echo(f"Total return: {report.total_return:.2%}")
    click.echo(f"Annual return: {report.annual_return:.2%}")
    click.echo(f"Max drawdown: {report.max_drawdown:.2%}")
    click.echo(f"Sharpe ratio: {report.sharpe_ratio:.2f}")
    click.echo(f"Win rate: {report.win_rate:.2%}")
    click.echo(f"Avg daily return: {report.avg_daily_return:.4%}")
    click.echo(f"Volatility: {report.volatility:.4%}")
    click.echo(f"Total trades: {report.total_trades}")
    click.echo(f"\nReport saved to: {report_path}")


@cli.command("selection-trading")
@click.option("--config", "config_path", default="config/selection_trading.yaml", show_default=True)
@click.option("--start", "start_date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", "end_date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--output-dir", "output_dir", default="runs/selection_trading", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def selection_trading_cmd(
    config_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    verbose: bool,
) -> None:
    """Run selection + RL trading pipeline (placeholder - not yet implemented)."""
    click.echo("Error: selection-trading pipeline not yet fully implemented.")
    click.echo("Current status: Phase 3 (selection module) completed.")
    click.echo("Remaining work: Phase 5-7 (RL integration + pipeline orchestration)")
    click.echo("\nYou can use these commands separately:")
    click.echo("  1. finsys selection run --start 2023-07-01 --end 2023-12-31")
    click.echo("  2. finsys train --data-file <selection_output>")
    click.echo("\nOr use the random training script:")
    click.echo("  python scripts/random_rl_train.py --stock-count 10 --train-days 60")


# ---------------------------------------------------------------------------
# run (orchestrated pipeline)
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option("--output", "output_dir", default="runs/latest", show_default=True)
@click.option(
    "-a",
    "--algo",
    "algorithm",
    type=click.Choice(["ppo", "sac", "td3"], case_sensitive=False),
    default=None,
)
@click.option("-t", "--timesteps", "total_timesteps", type=int, default=None)
@click.option("--enhanced", is_flag=True, default=False, help="Use sentiment-enhanced dataset.")
@click.option("--verbose", is_flag=True, default=False)
def run_cmd(
    config_path: str,
    output_dir: str,
    algorithm: str | None,
    total_timesteps: int | None,
    enhanced: bool,
    verbose: bool,
) -> None:
    """Full pipeline: fetch → [news → sentiment → fundamentals → fuse] → train → backtest."""
    import pandas as pd

    from finquant.data.pipeline import DataPipeline
    from finquant.features.fusion import fuse_datasets
    from finquant.training.trainer import Trainer

    config = load_config(config_path)
    if algorithm:
        config.training.algorithm = algorithm
    if total_timesteps:
        config.training.total_timesteps = total_timesteps

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch market data
    if verbose:
        click.echo("Step 1/4: Fetching market data ...")
    pipeline = DataPipeline(config)
    data_path = pipeline.fetch_and_save(out / "data")
    df = pd.read_parquet(data_path)

    enhanced_data_path = data_path
    sentiment_file: Path | None = None
    fundamental_file: Path | None = None

    if enhanced:
        # Step 1a: fetch news
        if verbose:
            click.echo("Step 1a: Fetching news ...")
        from finquant.data.news_fetcher import fetch_news_for_universe
        news_path = fetch_news_for_universe(
            tickers=config.stocks, output_dir=out / "news", verbose=verbose
        )

        # Step 1b: sentiment analysis
        if verbose:
            click.echo("Step 1b: Running sentiment analysis ...")
        from finquant.features.sentiment import SentimentProcessor
        sentiment_proc = SentimentProcessor(config.sentiment)
        sentiment_path = sentiment_proc.process_file(
            input_path=news_path,
            output_dir=out / "sentiment",
            verbose=verbose,
        )
        sentiment_file = sentiment_path

        # Step 1c: fundamental data
        if verbose:
            click.echo("Step 1c: Fetching fundamentals ...")
        from finquant.data.fundamental_fetcher import fetch_fundamentals_for_universe
        trading_dates = sorted(df["date"].unique().tolist())
        fundamental_path = fetch_fundamentals_for_universe(
            tickers=config.stocks,
            trading_dates=trading_dates,
            output_dir=out / "fundamentals",
            verbose=verbose,
        )
        fundamental_file = fundamental_path

        # Step 1d: fuse
        if verbose:
            click.echo("Step 1d: Fusing features ...")
        enhanced_data_path = out / "enhanced" / "dataset.parquet"
        enhanced_data_path.parent.mkdir(parents=True, exist_ok=True)
        fuse_datasets(
            market_df=df,
            sentiment_file=sentiment_file,
            fundamental_file=fundamental_file,
            output_path=enhanced_data_path,
            verbose=verbose,
        )
        df = pd.read_parquet(enhanced_data_path)

    # Step 2: train
    if verbose:
        click.echo(f"Step 2/4: Training {config.training.algorithm} ...")
    train_end = config.dates.train_end
    train_df = df[df["date"] <= train_end]
    test_df = df[df["date"] > train_end]

    trainer = Trainer(config)
    model_path = trainer.train(train_df=train_df, output_dir=out / "models")

    # Step 3: backtest
    if verbose:
        click.echo("Step 3/4: Running backtest ...")
    report = trainer.backtest(
        model_path=model_path,
        test_df=test_df,
        output_dir=out / "reports",
    )
    click.echo("Done.")
    import json
    click.echo(json.dumps(report.to_dict(), indent=2))


@cli.command("run-5min")
@click.option("--config", "config_path", default="config/csi500_5min.yaml", show_default=True)
@click.option("--output", "output_dir", default="runs/5min", show_default=True)
@click.option(
    "-a",
    "--algo",
    "algorithm",
    type=click.Choice(["ppo", "sac", "td3"], case_sensitive=False),
    default=None,
)
@click.option("-t", "--timesteps", "total_timesteps", type=int, default=None)
@click.option("--indicator-set", "indicator_set_id", default=None, help="Indicator set ID to train.")
@click.option("--all-indicator-sets", is_flag=True, default=False, help="Train all configured indicator sets.")
@click.option("--enhanced", is_flag=True, default=False, help="Use sentiment-enhanced dataset.")
@click.option("--verbose", is_flag=True, default=False)
def run_5min_cmd(
    config_path: str,
    output_dir: str,
    algorithm: str | None,
    total_timesteps: int | None,
    indicator_set_id: str | None,
    all_indicator_sets: bool,
    enhanced: bool,
    verbose: bool,
) -> None:
    """Full 5min pipeline: build dataset → train → backtest for one or all indicator sets."""
    import pandas as pd

    from finquant.config.settings import load_config
    from finquant.data.dataset import TrainingDatasetBuilder
    from finquant.data.pipeline import DataPipeline
    from finquant.features.fusion import fuse_datasets
    from finquant.features.indicator_sets import IndicatorSetRegistry
    from finquant.training.comparison import ComparisonAnalyzer
    from finquant.training.trainer import Trainer

    config = load_config(config_path)
    if algorithm:
        config.training.algorithm = algorithm
    if total_timesteps:
        config.training.total_timesteps = total_timesteps

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch 5min data
    if verbose:
        click.echo("Step 1/5: Fetching 5min market data ...")
    pipeline = DataPipeline(config)
    raw_df = pipeline.fetch()

    if verbose:
        click.echo(f"Fetched {len(raw_df)} rows, {raw_df['tic'].nunique()} tickers")

    # Build registry
    registry = IndicatorSetRegistry.from_configs(config.indicator_sets)
    builder = TrainingDatasetBuilder(config, registry=registry)

    # Determine which indicator sets to process
    if all_indicator_sets:
        iset_ids = registry.list_ids()
    else:
        iset_ids = [indicator_set_id or (config.indicator_sets[0].id if config.indicator_sets else None)]
        iset_ids = [i for i in iset_ids if i]

    for iset_id in iset_ids:
        run_out = out / iset_id
        run_out.mkdir(parents=True, exist_ok=True)

        # Step 2: build dataset
        if verbose:
            click.echo(f"Step 2/5: Building dataset for {iset_id} ...")
        dataset_path = builder.build(raw_df.copy(), indicator_set_id=iset_id, output_path=run_out / "dataset.parquet")
        df = pd.read_parquet(dataset_path)

        # Optional: fuse with sentiment/fundamentals
        if enhanced:
            if verbose:
                click.echo(f"Step 2a: Fusing features for {iset_id} ...")
            # Look for existing sentiment/fundamental files
            sentiment_file = out / "sentiment" / "sentiment.jsonl"
            fundamental_file = out / "fundamentals" / "fundamentals.jsonl"
            dataset_path = fuse_datasets(
                df,
                sentiment_file=sentiment_file if sentiment_file.exists() else None,
                fundamental_file=fundamental_file if fundamental_file.exists() else None,
                output_path=run_out / "enhanced_dataset.parquet",
                frequency="5min",
                verbose=verbose,
            )
            df = pd.read_parquet(dataset_path)

        # Step 3: train
        if verbose:
            click.echo(f"Step 3/5: Training {config.training.algorithm} for {iset_id} ...")
        train_end = config.dates.train_end
        train_df = df[df["date"] <= train_end]
        test_df = df[df["date"] > train_end]

        trainer = Trainer(config)
        model_path = trainer.train(
            train_df=train_df,
            output_dir=run_out / "models",
            indicator_set_id=iset_id,
        )

        # Step 4: backtest
        if verbose:
            click.echo(f"Step 4/5: Backtesting {iset_id} ...")
        report = trainer.backtest(
            model_path=model_path,
            test_df=test_df,
            output_dir=run_out / "reports",
            expected_obs_dim=trainer._last_train_obs_dim,
        )

        click.echo(f"Done: {iset_id} — sharpe={report.sharpe:.4f}")

    # Step 5: comparison report
    if len(iset_ids) > 1:
        if verbose:
            click.echo("Step 5/5: Generating comparison report ...")
        analyzer = ComparisonAnalyzer(out)
        comp_report = analyzer.run()
        comp_report.to_csv(out / "comparison.csv")
        comp_report.to_json(out / "comparison.json")
        click.echo(f"Comparison saved to {out / 'comparison.csv'}")


if __name__ == "__main__":
    cli()
