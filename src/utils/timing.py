"""
Simple wall-clock timing utility.

Used throughout the codebase to measure GP fit time, RF fit time,
and evaluation time — the three key metrics in the prototype spec.

Usage:
    from src.utils.timing import Timer

    with Timer("gp_fit") as t:
        gp.fit(X, y)
    print(f"GP fit took {t.elapsed:.3f}s")

    # Or without a context manager:
    t = Timer()
    t.start()
    result = expensive_fn()
    t.stop()
    print(t.elapsed)
"""

import time


class Timer:
    """
    Context manager that measures elapsed wall-clock time.

    Uses time.perf_counter() which is the highest-resolution timer
    available on the platform and is not affected by system clock
    adjustments (unlike time.time()).
    """

    def __init__(self, name: str = ""):
        """
        Args:
            name: Optional label for this timer (used in __repr__).
        """
        self.name = name
        self.elapsed: float = 0.0   # populated after stop() or __exit__
        self._start: float | None = None

    def start(self) -> "Timer":
        """Start timing.  Returns self for chaining."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop timing and store elapsed seconds.  Returns elapsed."""
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start().")
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed:.4f}s)"
