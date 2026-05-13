import click
import yaml
from pathlib import Path


_DEFAULT_CONFIG = str(Path(__file__).parent.parent / "config" / "default.yaml")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
@click.option("--config", default=str(_DEFAULT_CONFIG), show_default=True, help="Config file path")
@click.pass_context
def main(ctx: click.Context, config: str) -> None:
    """K-line ViT: K-line chart recognition and backtesting system."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["config_path"] = config


@main.command("build-dataset")
@click.option("--db-path", required=True, help="Path to hk_stocks.db")
@click.option("--output-dir", default="data/images", show_default=True, help="Image output directory")
@click.option("--codes", default=None, help="Comma-separated stock codes to process (default: all)")
@click.option("--limit", default=None, type=int, help="Process only the first N stocks (default: all)")
@click.pass_context
def build_dataset(ctx: click.Context, db_path: str, output_dir: str, codes: str | None, limit: int | None) -> None:
    """Build K-line image dataset from SQLite database."""
    from kline_vit.data.dataset import build_dataset_index
    from kline_vit.data.db_reader import DBReader

    config = ctx.obj["config"]
    config["data"]["db_path"] = db_path
    config["data"]["image_dir"] = output_dir

    filter_codes = codes.split(",") if codes else None

    reader = DBReader(db_path)
    all_codes = filter_codes if filter_codes else reader.get_all_codes()
    if limit is not None:
        all_codes = all_codes[:limit]
        filter_codes = all_codes
    click.echo(f"DB: {db_path}")
    click.echo(f"Output: {output_dir}")
    click.echo(f"Stocks to process: {len(all_codes)}")

    for split in ("train", "val", "test"):
        click.echo(f"\n[{split}] Building split...")
        df = build_dataset_index(
            db_path, output_dir, config, split,
            filter_codes=filter_codes,
            progress_callback=lambda code, done, total, n_records: click.echo(
                f"  [{split}] {done}/{total} {code:>10}  records so far: {n_records}"
            ),
        )
        out_csv = Path(output_dir).parent / f"dataset_{split}.csv"
        df.to_csv(out_csv, index=False)
        if len(df):
            buy = (df["label"] == 1).sum()
            click.echo(f"  [{split}] done — {len(df)} samples (buy={buy}, hold/sell={len(df)-buy}) → {out_csv}")
        else:
            click.echo(f"  [{split}] done — 0 samples → {out_csv}")

    click.echo("\nDataset build complete.")


@main.command("train")
@click.option("--train-csv", required=True, help="Path to dataset_train.csv")
@click.option("--val-csv", required=True, help="Path to dataset_val.csv")
@click.option("--test-csv", required=True, help="Path to dataset_test.csv")
@click.pass_context
def train(ctx: click.Context, train_csv: str, val_csv: str, test_csv: str) -> None:
    """Fine-tune ViT model on K-line image dataset."""
    import torch
    from torch.utils.data import DataLoader
    from kline_vit.data.dataset import KlineDataset
    from kline_vit.model.vit_finetuner import ViTFineTuner

    config = ctx.obj["config"]

    train_ds = KlineDataset(train_csv)
    val_ds = KlineDataset(val_csv)
    test_ds = KlineDataset(test_csv)

    click.echo(f"Train CSV:  {train_csv}  ({len(train_ds)} samples)")
    click.echo(f"Val CSV:    {val_csv}  ({len(val_ds)} samples)")
    click.echo(f"Test CSV:   {test_csv}  ({len(test_ds)} samples)")

    bs = config["model"]["batch_size"]
    click.echo(f"Batch size: {bs}")
    click.echo("")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    finetuner = ViTFineTuner(config)
    class_weights = train_ds.get_class_weights()
    click.echo(f"Class weights: hold={class_weights[0]:.3f}  buy={class_weights[1]:.3f}")
    click.echo("")
    finetuner.train(train_loader, val_loader, class_weights=class_weights)

    click.echo("\nEvaluating on test set...")
    metrics = finetuner.evaluate(test_loader)
    cm = metrics["confusion_matrix"]
    click.echo(f"  Accuracy:  {metrics['accuracy']:.4f}")
    click.echo(f"  Precision: {metrics['precision']:.4f}")
    click.echo(f"  Recall:    {metrics['recall']:.4f}")
    click.echo(f"  F1:        {metrics['f1']:.4f}")
    click.echo(f"  Confusion Matrix (rows=actual, cols=predicted):")
    click.echo(f"             Hold   Buy")
    click.echo(f"    Hold  {cm[0][0]:>6}  {cm[0][1]:>6}")
    click.echo(f"    Buy   {cm[1][0]:>6}  {cm[1][1]:>6}")


@main.command("backtest")
@click.option("--model-path", required=True, help="Path to trained model .pth file")
@click.option("--db-path", required=True, help="Path to hk_stocks.db")
@click.pass_context
def backtest(ctx: click.Context, model_path: str, db_path: str) -> None:
    """Run Backtrader backtest using ViT model signals."""
    from kline_vit.model.inference import InferenceEngine
    from kline_vit.backtest.runner import BacktestRunner

    config = ctx.obj["config"]
    config["data"]["db_path"] = db_path

    engine = InferenceEngine(model_path, config)
    runner = BacktestRunner(config)
    report = runner.run(engine, db_path)

    click.echo("\n=== Backtest Report ===")
    click.echo(f"  Annual Return:    {report.annual_return:.2%}")
    click.echo(f"  Max Drawdown:     {report.max_drawdown:.2%}")
    click.echo(f"  Sharpe Ratio:     {report.sharpe_ratio:.2f}")
    click.echo(f"  Win Rate:         {report.win_rate:.2%}")
    click.echo(f"  Profit Factor:    {report.profit_factor:.2f}")
    click.echo(f"  Total Trades:     {report.total_trades}")
    click.echo(f"  Benchmark Return: {report.benchmark_return:.2%}")
    click.echo(f"  Excess Return:    {report.excess_return:.2%}")


@main.command("infer")
@click.option("--model-path", required=True, help="Path to trained model .pth file")
@click.option("--image-path", required=True, help="Path to K-line image PNG")
@click.pass_context
def infer(ctx: click.Context, model_path: str, image_path: str) -> None:
    """Run inference on a single K-line image."""
    from kline_vit.model.inference import InferenceEngine

    config = ctx.obj["config"]
    engine = InferenceEngine(model_path, config)
    result = engine.predict_single(image_path)

    click.echo(f"Image:            {result.image_path}")
    click.echo(f"Buy Probability:  {result.buy_probability:.4f}")
    click.echo(f"Signal:           {'BUY' if result.label == 1 else 'HOLD/SELL'}")
    click.echo(f"Inference Time:   {result.inference_time_ms:.1f}ms")
