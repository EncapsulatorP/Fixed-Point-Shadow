"""Bounded Formal System (BFS) framework.

Implements Definitions 1-6 from the README:

  Def 1  BFS: a triple S = (X, F, d)
  Def 2  Fixed points and cycles
  Def 3  Basin of attraction  B(x₀) = { x | d(Fⁿ(x), x₀) → 0 }
  Def 4  Decidable / tractable core  D(S)
  Def 5  Shadow / hard region        U(S) = X \\ D(S)
  Def 6  Boundary wall               ∂S = closure(D(S)) ∩ closure(U(S))

All functions operate on discrete, finite samples of X.  The continuous
versions (closure, boundary) are approximated on the supplied grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Definition 1 — Bounded Formal System
# ---------------------------------------------------------------------------

@dataclass
class BFS:
    """Bounded Formal System  S = (X, F, d).

    Parameters
    ----------
    F:
        Deterministic update rule  F : ℝⁿ → ℝⁿ  (or any numpy-compatible
        callable operating on a single point represented as a float or array).
    d:
        Metric  d : X × X → ℝ≥0.  Defaults to the Euclidean distance when
        left as *None*.

    Notes
    -----
    X itself is not stored in the dataclass because it can be infinite or
    implicitly defined.  Pass explicit *candidates* to the helper functions
    below.
    """

    F: Callable
    d: Callable | None = None

    def __post_init__(self) -> None:
        if self.d is None:
            def _default_d(a, b) -> float:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return abs(float(a) - float(b))
                return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
            self.d = _default_d


# ---------------------------------------------------------------------------
# Definition 2 — Fixed points and cycles
# ---------------------------------------------------------------------------

def is_fixed_point(bfs: BFS, x, tol: float = 1e-10) -> bool:
    """Return True if  d(F(x), x) < tol  (approximate fixed point)."""
    return bfs.d(bfs.F(x), x) < tol


def find_fixed_points(bfs: BFS, candidates: Iterable, tol: float = 1e-10) -> list:
    """Return those candidates that are approximate fixed points of F."""
    return [x for x in candidates if is_fixed_point(bfs, x, tol)]


def orbit(bfs: BFS, x, n: int) -> list:
    """Return the orbit  [x, F(x), F²(x), …, Fⁿ(x)]."""
    pts = [x]
    for _ in range(n):
        x = bfs.F(x)
        pts.append(x)
    return pts


def detect_cycle(bfs: BFS, x, max_iter: int = 10_000, tol: float = 1e-9) -> list | None:
    """Detect a periodic orbit starting from x using Floyd's algorithm.

    Returns the list of distinct cycle points, or *None* if no cycle is
    found within *max_iter* steps.
    """
    slow = x
    fast = bfs.F(x)
    for _ in range(max_iter):
        if bfs.d(slow, fast) < tol:
            break
        slow = bfs.F(slow)
        fast = bfs.F(bfs.F(fast))
    else:
        return None

    # Collect the cycle
    cycle = [slow]
    cur = bfs.F(slow)
    for _ in range(max_iter):
        if bfs.d(cur, slow) < tol:
            break
        cycle.append(cur)
        cur = bfs.F(cur)
    return cycle


# ---------------------------------------------------------------------------
# Definition 3 — Basin of attraction
# ---------------------------------------------------------------------------

def in_basin(bfs: BFS, x, attractor, n_iter: int = 1000, tol: float = 1e-8) -> bool:
    """Test whether x lies in the basin of attraction of *attractor*.

    Iterates F up to *n_iter* times; returns True as soon as
    d(Fᵏ(x), attractor) < tol for some k ≤ n_iter.
    """
    cur = x
    for _ in range(n_iter):
        if bfs.d(cur, attractor) < tol:
            return True
        cur = bfs.F(cur)
    return bfs.d(cur, attractor) < tol


def basin_of_attraction(bfs: BFS, attractor, candidates: Iterable,
                         n_iter: int = 1000, tol: float = 1e-8) -> list:
    """Compute  B(attractor) = { x ∈ candidates | Fⁿ(x) → attractor }.

    Implements Definition 3 from the README.
    """
    return [x for x in candidates if in_basin(bfs, x, attractor, n_iter, tol)]


# ---------------------------------------------------------------------------
# Definitions 4 & 5 — Decidable core D(S) and Shadow U(S)
# ---------------------------------------------------------------------------

def decidable_core(bfs: BFS, candidates: Iterable, attractors: Iterable,
                   n_iter: int = 1000, tol: float = 1e-8) -> list:
    """Compute D(S): points that converge to some known attractor.

    A point is placed in D(S) when the orbit can be *certified* to reach a
    known attracting set within *n_iter* steps.  Points that do not converge
    are placed in U(S) by :func:`shadow_region`.

    Implements Definition 4 from the README (operational / finite-sample
    version).
    """
    attractor_list = list(attractors)
    core = []
    for x in candidates:
        for att in attractor_list:
            if in_basin(bfs, x, att, n_iter, tol):
                core.append(x)
                break
    return core


def shadow_region(candidates: Iterable, core: Iterable) -> list:
    """Compute U(S) = X \\ D(S).

    Returns those candidates that are *not* in the core.  Implements
    Definition 5 from the README.
    """
    core_list = list(core)
    return [x for x in candidates if not any(
        np.allclose(x, c) for c in core_list
    )]


# ---------------------------------------------------------------------------
# Definition 6 — Boundary wall ∂S  (finite-sample approximation)
# ---------------------------------------------------------------------------

def boundary_wall(candidates: np.ndarray, core_mask: np.ndarray,
                  epsilon: float | None = None) -> np.ndarray:
    """Approximate  ∂S = closure(D(S)) ∩ closure(U(S)).

    Parameters
    ----------
    candidates:
        Array of shape (N,) or (N, d) listing all sample points.
    core_mask:
        Boolean array of length N; True = point is in D(S).
    epsilon:
        Neighbourhood radius used to define "closure" on the finite sample.
        Defaults to 2× the median pairwise spacing.

    Returns
    -------
    Boolean mask of length N; True = point is on the boundary.

    Notes
    -----
    For *complex-analytic* F the boundary coincides with the Julia set.
    In other settings this is an *analogy* (see §5 of the README).
    """
    candidates = np.asarray(candidates, float)
    if candidates.ndim == 1:
        candidates = candidates[:, None]

    if epsilon is None:
        diffs = np.abs(np.diff(candidates[:, 0]))
        epsilon = 2.0 * float(np.median(diffs[diffs > 0])) if diffs.size else 0.1

    n = len(candidates)
    on_boundary = np.zeros(n, dtype=bool)

    for i in range(n):
        dists = np.linalg.norm(candidates - candidates[i], axis=1)
        neighbours = dists < epsilon
        neighbours[i] = False
        if neighbours.any():
            # i is on the boundary if its neighbourhood contains points from
            # both D(S) and U(S)
            neighbour_labels = core_mask[neighbours]
            if neighbour_labels.any() and not neighbour_labels.all():
                on_boundary[i] = True

    return on_boundary
