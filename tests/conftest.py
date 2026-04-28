from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "tic": ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
            "open": [10.0, 12.0, 10.5, 12.2],
            "high": [10.2, 12.4, 10.7, 12.5],
            "low": [9.8, 11.8, 10.1, 12.0],
            "close": [10.1, 12.1, 10.6, 12.3],
            "volume": [10000.0, 12000.0, 15000.0, 13000.0],
        }
    )


@pytest.fixture
def sample_texts() -> list[dict[str, str]]:
    return [
        {"date": "2025-01-02", "tic": "000001.SZ", "text": "公司发布利好公告，利润大幅增长"},
        {"date": "2025-01-03", "tic": "600000.SH", "text": "监管处罚导致市场情绪偏弱"},
    ]
