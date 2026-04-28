from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def retry(max_retries: int = 3, base_delay: float = 1.0, factor: float = 2.0) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_error: Exception | None = None
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(delay)
                    delay *= factor
            if last_error is not None:
                raise last_error
            raise RuntimeError("retry failed without exception")

        return wrapper  # type: ignore[return-value]

    return decorator
