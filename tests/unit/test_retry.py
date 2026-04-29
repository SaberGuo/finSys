from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finquant.utils.retry import retry


class TestRetryDecorator:
    def test_returns_immediately_on_success(self) -> None:
        mock = MagicMock(return_value=42)

        @retry(max_retries=3)
        def func() -> int:
            return mock()

        result = func()
        assert result == 42
        assert mock.call_count == 1

    def test_retries_then_succeeds(self) -> None:
        mock = MagicMock(side_effect=[RuntimeError("fail1"), RuntimeError("fail2"), 99])

        @retry(max_retries=3, base_delay=0.01)
        def func() -> int:
            return mock()

        with patch("finquant.utils.retry.time.sleep") as mock_sleep:
            result = func()

        assert result == 99
        assert mock.call_count == 3
        mock_sleep.assert_called()

    def test_raises_after_all_retries_exhausted(self) -> None:
        mock = MagicMock(side_effect=RuntimeError("always fails"))

        @retry(max_retries=2, base_delay=0.01)
        def func() -> int:
            return mock()

        with patch("finquant.utils.retry.time.sleep"):
            with pytest.raises(RuntimeError, match="always fails"):
                func()

        assert mock.call_count == 2

    def test_backoff_increases_delay(self) -> None:
        mock = MagicMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), 1])

        @retry(max_retries=3, base_delay=0.1, factor=2.0)
        def func() -> int:
            return mock()

        with patch("finquant.utils.retry.time.sleep") as mock_sleep:
            func()

        calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert calls == [0.1, 0.2]
