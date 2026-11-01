"""Execution strategies for bulk sampling, and how one gets chosen.

The same field evaluation has a very different optimum at different sizes. A
single point wants no machinery at all -- the numpy setup cost dominates. A
million points want one vectorised pass. A billion points want blocking,
because the intermediate arrays no longer fit in cache, let alone RAM. And
work that is genuinely per-pixel on a GPU should never come back to the CPU
at all.

So the strategy is a parameter, not a hardcoded choice::

    sample_field(name, "density", bounds=..., resolution=512)              # auto
    sample_field(name, "density", bounds=..., resolution=512, how="vector")
    sample_field(name, "density", bounds=..., resolution=512, how="chunked")

    plan(name, "density", points=512**3)      # what would auto pick, and why

Strategies
----------
``scalar``    One Python call per point. Zero setup, no allocation beyond the
              result. Fastest below roughly a thousand points and the only
              option for a field model that is not vectorisable.
``vector``    One numpy pass over the whole grid. The default for ordinary
              work; 100-1000x faster than ``scalar`` per point.
``chunked``   ``vector`` over blocks sized to stay inside cache, writing into
              a preallocated output. Same arithmetic, bounded peak memory --
              the difference between a 4096^3 volume running and dying.
``parallel``  ``chunked`` with the blocks spread over a worker pool. Only
              worth its process-startup and pickling cost for large grids.

GPU
---
This module does not execute anything on a GPU, deliberately. Shipping a CUDA
dependency to run arithmetic that the caller's engine is already set up to run
per-pixel would be the wrong trade. What ships instead is the *formula*: see
`periodica.engine.shader`, which emits a channel's resolved expression as
HLSL, GLSL or WGSL together with the uniform block it needs, so the work
happens on-device with no CPU round trip and no runtime dependency here.

Auto-selection policy
---------------------
Thresholds are deliberate, documented and overridable, not magic numbers
buried in a branch::

    < 1e3 points              scalar    numpy setup cost dominates
    < 8e6 points              vector    fits comfortably in memory
    < 1e8 points              chunked   bound peak memory, keep one core
    >= 1e8 points, workers>1  parallel  amortises pool startup

Measure before overriding: the crossover moves with the field model's cost per
point, and `plan()` reports the estimate so a caller can compare.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Iterator, NamedTuple, Optional, Sequence, Tuple

import numpy as np


class Strategy(str, Enum):
    """How a bulk sample is executed."""

    AUTO = "auto"
    SCALAR = "scalar"
    VECTOR = "vector"
    CHUNKED = "chunked"
    PARALLEL = "parallel"

    def __str__(self) -> str:
        return self.value


#: Point counts at which auto-selection changes strategy. See the module
#: docstring for the reasoning; override per call via `thresholds=`.
DEFAULT_THRESHOLDS = {
    "scalar_max": 1_000,
    "vector_max": 8_000_000,
    "chunked_max": 100_000_000,
}

#: Target bytes per chunk for `chunked`/`parallel`. 8 MiB of float64 keeps a
#: block and its working set inside a typical L2/L3 slice.
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


def coerce_strategy(how: Optional[str | Strategy]) -> Strategy:
    """Normalise a strategy argument, with a helpful error for typos."""
    if how is None:
        return Strategy.AUTO
    if isinstance(how, Strategy):
        return how
    try:
        return Strategy(str(how).lower())
    except ValueError:
        raise ValueError(
            f"Unknown strategy {how!r}. Choose from {[s.value for s in Strategy]}."
        ) from None


def worker_default() -> int:
    """Worker count for `parallel`: the CPU count, capped to keep it sane."""
    return max(1, min(os.cpu_count() or 1, 32))


class Plan(NamedTuple):
    """What a bulk sample is about to do, before it does it.

    Returned by `plan()` so a caller can decide -- or a downstream app can
    show a progress estimate -- without running anything.
    """

    strategy: Strategy
    points: int
    chunks: int
    chunk_points: int
    bytes_peak: int
    workers: int
    reason: str

    def as_dict(self) -> dict:
        return dict(self._asdict(), strategy=self.strategy.value)

    def __str__(self) -> str:
        return (
            f"{self.strategy}: {self.points:,} points in {self.chunks:,} chunk(s) "
            f"of {self.chunk_points:,}, peak ~{self.bytes_peak / 1024**2:.1f} MiB"
            f"{f', {self.workers} workers' if self.workers > 1 else ''} "
            f"({self.reason})"
        )


def choose(
    points: int,
    *,
    how: Optional[str | Strategy] = None,
    workers: Optional[int] = None,
    thresholds: Optional[dict] = None,
    vectorisable: bool = True,
) -> Tuple[Strategy, str]:
    """Pick a strategy for `points`, returning it with the reason.

    An explicit `how` is honoured except where it cannot work: a field model
    that is not vectorisable falls back to `scalar` and says so, rather than
    failing deep inside a numpy call.
    """
    requested = coerce_strategy(how)
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    n_workers = workers if workers is not None else worker_default()

    if requested is not Strategy.AUTO:
        if requested is not Strategy.SCALAR and not vectorisable:
            return Strategy.SCALAR, (
                f"{requested} requested but this field model is not "
                f"vectorisable; fell back to scalar"
            )
        if requested is Strategy.PARALLEL and n_workers < 2:
            return Strategy.CHUNKED, (
                "parallel requested but only one worker is available; "
                "chunked is the same arithmetic without the pool"
            )
        return requested, f"{requested} requested explicitly"

    if not vectorisable:
        return Strategy.SCALAR, "field model is not vectorisable"
    if points < limits["scalar_max"]:
        return Strategy.SCALAR, (
            f"{points:,} points is below {limits['scalar_max']:,}; "
            f"numpy setup cost would dominate"
        )
    if points < limits["vector_max"]:
        return Strategy.VECTOR, (
            f"{points:,} points fits comfortably in memory as one pass"
        )
    if points < limits["chunked_max"] or n_workers < 2:
        return Strategy.CHUNKED, (
            f"{points:,} points needs blocking to bound peak memory"
        )
    return Strategy.PARALLEL, (
        f"{points:,} points amortises a {n_workers}-worker pool"
    )


def plan(
    points: int,
    *,
    how: Optional[str | Strategy] = None,
    workers: Optional[int] = None,
    thresholds: Optional[dict] = None,
    vectorisable: bool = True,
    itemsize: int = 8,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Plan:
    """Describe what sampling `points` values would do, without doing it."""
    strategy, reason = choose(
        points, how=how, workers=workers, thresholds=thresholds,
        vectorisable=vectorisable,
    )
    n_workers = (workers if workers is not None else worker_default()) \
        if strategy is Strategy.PARALLEL else 1

    if strategy in (Strategy.CHUNKED, Strategy.PARALLEL):
        chunk_points = max(1, chunk_bytes // max(1, itemsize))
        chunks = max(1, -(-points // chunk_points))
        # Output always lands in full; only the working set is blocked.
        peak = points * itemsize + chunk_points * itemsize * 3 * n_workers
    elif strategy is Strategy.VECTOR:
        chunk_points, chunks = points, 1
        # Result plus the coordinate arrays the pass needs alongside it.
        peak = points * itemsize * 4
    else:
        chunk_points, chunks = 1, points
        peak = points * itemsize

    return Plan(strategy, points, chunks, chunk_points, int(peak), n_workers, reason)


# ── Chunk iteration ─────────────────────────────────────────────────────

def chunk_slices(total: int, chunk_points: int) -> Iterator[slice]:
    """Contiguous slices covering `total` in `chunk_points`-sized blocks."""
    if chunk_points < 1:
        raise ValueError("chunk_points must be >= 1")
    for start in range(0, total, chunk_points):
        yield slice(start, min(start + chunk_points, total))


def run_chunked(
    total: int,
    kernel: Callable[[slice], np.ndarray],
    *,
    out: np.ndarray,
    chunk_points: int,
) -> np.ndarray:
    """Fill `out` by calling `kernel` per block. Bounded peak memory."""
    flat = out.reshape(-1)
    for window in chunk_slices(total, chunk_points):
        flat[window] = kernel(window)
    return out


def run_parallel(
    total: int,
    kernel: Callable[[slice], np.ndarray],
    *,
    out: np.ndarray,
    chunk_points: int,
    workers: int,
) -> np.ndarray:
    """`run_chunked` with blocks spread over a thread pool.

    Threads, not processes: the per-block work is numpy arithmetic that
    releases the GIL, and threads avoid pickling a kernel closure and copying
    every block through a pipe. A process pool would need the kernel to be
    importable at module scope, which rules out the closures the grid sampler
    builds per call.
    """
    from concurrent.futures import ThreadPoolExecutor

    flat = out.reshape(-1)
    windows = list(chunk_slices(total, chunk_points))
    if len(windows) == 1 or workers < 2:
        flat[windows[0]] = kernel(windows[0])
        return out

    def fill(window: slice) -> None:
        flat[window] = kernel(window)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() forces every future so exceptions surface here, not silently.
        list(pool.map(fill, windows))
    return out


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_THRESHOLDS",
    "Plan",
    "Strategy",
    "choose",
    "chunk_slices",
    "coerce_strategy",
    "plan",
    "run_chunked",
    "run_parallel",
    "worker_default",
]
