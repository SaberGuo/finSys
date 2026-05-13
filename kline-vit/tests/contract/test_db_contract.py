import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from kline_vit.data.db_reader import DBReader

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "hk_stocks.db"


@pytest.fixture
def db_reader():
    if not DB_PATH.exists():
        pytest.skip(f"Database not found at {DB_PATH}")
    return DBReader(str(DB_PATH))


def test_daily_data_schema(db_reader):
    df = db_reader.get_daily_data("hk.00001", "2024-01-31", n_days=60)
    assert set(df.columns) == {"date", "open", "high", "low", "close", "volume", "amount"}
    assert df["date"].dtype == object  # str
    assert df["close"].dtype == float
    assert len(df) <= 60
    assert df["date"].is_monotonic_increasing


def test_empty_dataframe_on_missing_code(db_reader):
    df = db_reader.get_daily_data("hk.NONEXISTENT", "2024-01-31", n_days=60)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_row_count_respects_n_days(db_reader):
    df = db_reader.get_daily_data("hk.00001", "2024-12-31", n_days=10)
    assert len(df) <= 10


def test_date_monotonic_increasing(db_reader):
    df = db_reader.get_daily_data("hk.00001", "2024-06-30", n_days=30)
    if len(df) > 1:
        assert df["date"].is_monotonic_increasing


def test_get_all_codes_returns_list(db_reader):
    codes = db_reader.get_all_codes()
    assert isinstance(codes, list)
    assert len(codes) > 0
    assert all(isinstance(c, str) for c in codes)


def test_get_date_range(db_reader):
    min_d, max_d = db_reader.get_date_range("hk.00001")
    assert min_d <= max_d
    assert len(min_d) == 10  # YYYY-MM-DD


def test_get_date_range_missing_code(db_reader):
    with pytest.raises(KeyError):
        db_reader.get_date_range("hk.NONEXISTENT")


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        DBReader("/nonexistent/path/db.sqlite")
