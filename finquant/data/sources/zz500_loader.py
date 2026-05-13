"""Load ZZ500 stock list from database."""
import sqlite3
from pathlib import Path


def load_zz500_stocks(db_path: Path | str) -> list[str]:
    """Load all unique stock codes from zz500_data.db.

    Args:
        db_path: Path to zz500_data.db

    Returns:
        List of stock codes in format "XXXXXX.SH" or "XXXXXX.SZ"

    Raises:
        FileNotFoundError: If database doesn't exist
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT DISTINCT code FROM daily_data ORDER BY code"
        )
        stocks = [row[0] for row in cursor.fetchall()]

    return stocks
