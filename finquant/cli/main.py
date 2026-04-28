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
    "--algo",
    "algorithm",
    type=click.Choice(["ppo", "sac", "td3"], case_sensitive=False),
    default=None,
    help="Override training algorithm from config.",
)
@click.option("--timesteps", "total_timesteps", type=int, default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def train_cmd(
    config_path: str,
    data_file: str,
    output_dir: str,
    algorithm: str | None,
    total_timesteps: int | None,
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

    if dry_run:
        click.echo(
            f"[dry-run] Would train {config.training.algorithm} "
            f"for {config.training.total_timesteps} steps on {data_file}"
        )
        return

    df = pd.read_parquet(data_file)
    if verbose:
        click.echo(
            f"Training {config.training.algorithm} on {df['tic'].nunique()} stocks "
            f"× {df['date'].nunique()} days ..."
        )

    trainer = Trainer(config)
    model_path = trainer.train(train_df=df, output_dir=Path(output_dir))
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
    required=True,
    help="Sentiment JSONL file (output of `sentiment analyze`).",
)
@click.option("--output", "output_file", default="data/enhanced/dataset.parquet", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def fuse_cmd(
    config_path: str,
    market_file: str,
    sentiment_file: str,
    output_file: str,
    verbose: bool,
) -> None:
    """Fuse market data with sentiment features to create enhanced dataset."""
    import pandas as pd

    from finquant.features.fusion import fuse_datasets

    market_df = pd.read_parquet(market_file)
    out_path = fuse_datasets(
        market_df=market_df,
        sentiment_file=Path(sentiment_file),
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


# ---------------------------------------------------------------------------
# run (orchestrated pipeline)
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--config", "config_path", default="config/default.yaml.example", show_default=True)
@click.option("--output", "output_dir", default="runs/latest", show_default=True)
@click.option(
    "--algo",
    "algorithm",
    type=click.Choice(["ppo", "sac", "td3"], case_sensitive=False),
    default=None,
)
@click.option("--timesteps", "total_timesteps", type=int, default=None)
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
    """Full pipeline: fetch → train → backtest (orchestrator)."""
    import pandas as pd

    from finquant.data.pipeline import DataPipeline
    from finquant.training.trainer import Trainer

    config = load_config(config_path)
    if algorithm:
        config.training.algorithm = algorithm
    if total_timesteps:
        config.training.total_timesteps = total_timesteps

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch data
    if verbose:
        click.echo("Step 1/3: Fetching market data ...")
    pipeline = DataPipeline(config)
    data_path = pipeline.fetch_and_save(out / "data")
    df = pd.read_parquet(data_path)

    # Step 2: train
    if verbose:
        click.echo(f"Step 2/3: Training {config.training.algorithm} ...")
    train_end = config.dates.train_end
    train_df = df[df["date"] <= train_end]
    test_df = df[df["date"] > train_end]

    trainer = Trainer(config)
    model_path = trainer.train(train_df=train_df, output_dir=out / "models")

    # Step 3: backtest
    if verbose:
        click.echo("Step 3/3: Running backtest ...")
    report = trainer.backtest(
        model_path=model_path,
        test_df=test_df,
        output_dir=out / "reports",
    )
    click.echo("Done.")
    import json
    click.echo(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    cli()
