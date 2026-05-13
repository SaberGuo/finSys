"""Unit tests for finquant.data.news_fetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finquant.data.news_fetcher import fetch_news_for_universe, fetch_stock_news


class TestFetchStockNews:
    @patch("akshare.stock_news_em")
    def test_fetch_single_stock(self, mock_em: MagicMock) -> None:
        mock_em.return_value = pd.DataFrame(
            [
                {
                    "关键词": "000009",
                    "新闻标题": "Title 1",
                    "新闻内容": "Content 1",
                    "发布时间": "2024-01-02 10:00:00",
                    "文章来源": "East Money",
                    "新闻链接": "http://example.com/1",
                },
                {
                    "关键词": "000009",
                    "新闻标题": "Title 2",
                    "新闻内容": "Content 2",
                    "发布时间": "2024-01-03 11:00:00",
                    "文章来源": "East Money",
                    "新闻链接": "http://example.com/2",
                },
            ]
        )
        records = fetch_stock_news("000009.SZ", max_pages=1)
        assert len(records) == 2
        assert records[0]["date"] == "2024-01-02"
        assert records[0]["tic"] == "000009.SZ"
        assert records[0]["title"] == "Title 1"
        assert records[0]["content"] == "Content 1"

    @patch("akshare.stock_news_em")
    def test_empty_result(self, mock_em: MagicMock) -> None:
        mock_em.return_value = pd.DataFrame()
        records = fetch_stock_news("000009.SZ", max_pages=1)
        assert records == []

    @patch("akshare.stock_news_em")
    def test_api_error_graceful(self, mock_em: MagicMock) -> None:
        mock_em.side_effect = RuntimeError("API error")
        records = fetch_stock_news("000009.SZ", max_pages=1)
        assert records == []


class TestFetchNewsForUniverse:
    @patch("akshare.stock_news_em")
    def test_fetch_and_save(self, mock_em: MagicMock, tmp_path: pytest.TempPathFactory) -> None:
        mock_em.return_value = pd.DataFrame(
            [
                {
                    "关键词": "000009",
                    "新闻标题": "T1",
                    "新闻内容": "C1",
                    "发布时间": "2024-01-02 10:00:00",
                    "文章来源": "EM",
                    "新闻链接": "",
                }
            ]
        )
        out = fetch_news_for_universe(
            tickers=["000009.SZ", "600004.SH"],
            output_dir=tmp_path,
            max_pages=1,
            sleep_seconds=0,
            verbose=False,
        )
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        import json
        rec = json.loads(lines[0])
        assert rec["tic"] == "000009.SZ"

    @patch("akshare.stock_news_em")
    def test_deduplication(self, mock_em: MagicMock, tmp_path: pytest.TempPathFactory) -> None:
        # Same title for both tickers should be deduped per ticker
        mock_em.return_value = pd.DataFrame(
            [
                {
                    "关键词": "000009",
                    "新闻标题": "Same Title",
                    "新闻内容": "C1",
                    "发布时间": "2024-01-02 10:00:00",
                    "文章来源": "EM",
                    "新闻链接": "",
                }
            ]
        )
        out = fetch_news_for_universe(
            tickers=["000009.SZ"],
            output_dir=tmp_path,
            max_pages=2,
            sleep_seconds=0,
            verbose=False,
        )
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        # 2 pages × 1 record each, but deduped by (date, tic, title) → 1
        assert len(lines) == 1
