"""Forward / Reverse map pair and adjunction-failure helpers.

Covers §2 of the README:

  L : C → C   forward evolution / compilation / coarse-graining
  R : C → C   reverse / reconstruction map

If L ⊣ R (L is left-adjoint to R), the unit  η : Id_C → R ∘ L  is a
natural transformation, meaning  R(L(x))  always contains at least as
much "information" as x.

When L is many-to-one (non-injective), any R must be multi-valued or
approximate.  This module formalises and measures that failure.

Key correction from the README (§2)
------------------------------------
"Many-to-one" forward behaviour is *sufficient* for information loss, but
information loss is **not** implied by positive entropy alone — invertible
maps can have positive topological entropy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Forward / Reverse pair
# ---------------------------------------------------------------------------

@dataclass
class ForwardMap:
    """A forward map  L : X → Y  with metadata.

    Parameters
    ----------
    apply:
        Callable implementing L(x).
    is_injective:
        Set to True only if L is provably injective (one-to-one).
        The default *None* means "unknown".
    """

    apply: Callable
    is_injective: bool | None = None

    def __call__(self, x):
        return self.apply(x)


@dataclass
class ReverseMap:
    """A reverse / reconstruction map  R : Y → X  with metadata.

    Parameters
    ----------
    apply:
        Callable implementing R(y), returning a *list* of candidates in X
        (multi-valued in general).
    is_exact:
        True if R is an exact left-inverse: L(R(y)[0]) == y for all y.
    """

    apply: Callable
    is_exact: bool = False

    def __call__(self, y) -> list:
        result = self.apply(y)
        if not isinstance(result, list):
            result = [result]
        return result


# ---------------------------------------------------------------------------
# Unit of the adjunction  η(x) = R(L(x))
# ---------------------------------------------------------------------------

def unit_eta(L: ForwardMap, R: ReverseMap, x) -> list:
    """Compute the unit of the (attempted) adjunction: η(x) = R(L(x)).

    In a true adjunction every candidate in R(L(x)) is "close to" x.
    When L is non-injective, R(L(x)) may contain many pre-images; x is
    just one of them.

    Returns
    -------
    List of pre-image candidates returned by R ∘ L.
    """
    return R(L(x))


# ---------------------------------------------------------------------------
# Adjunction failure diagnostics
# ---------------------------------------------------------------------------

def fibre_size(L: ForwardMap, candidates, tol: float = 1e-10) -> dict:
    """Compute |L⁻¹(y)| for each distinct image y in L(candidates).

    A fibre of size > 1 is direct evidence that L is many-to-one (non-injective)
    and therefore information-discarding.

    Parameters
    ----------
    L:
        Forward map.
    candidates:
        Iterable of points to probe.
    tol:
        Two images are considered equal when |L(a) - L(b)| < tol.

    Returns
    -------
    dict mapping each (rounded) image value to its multiplicity.
    """
    images: list = []
    for x in candidates:
        y = L(x)
        y_scalar = float(y) if np.isscalar(y) else float(np.asarray(y).ravel()[0])
        images.append((y_scalar, x))

    # Group by image (up to tol)
    images.sort(key=lambda p: p[0])
    fibre: dict[float, list] = {}
    for y_val, x in images:
        placed = False
        for key in fibre:
            if abs(key - y_val) < tol:
                fibre[key].append(x)
                placed = True
                break
        if not placed:
            fibre[y_val] = [x]

    return {y: len(xs) for y, xs in fibre.items()}


def reconstruction_error(L: ForwardMap, R: ReverseMap, x,
                          d: Callable | None = None) -> float:
    """Measure the best-case reconstruction error  min_{x̂ ∈ R(L(x))} d(x̂, x).

    A non-zero minimum shows that the adjunction unit η(x) ≠ x, i.e., the
    round-trip  x → L(x) → R(L(x))  does not recover x exactly.

    Parameters
    ----------
    L, R:
        Forward and reverse maps.
    x:
        Point to test.
    d:
        Metric.  Defaults to Euclidean distance.

    Returns
    -------
    Minimum distance between x and any candidate in R(L(x)).
    """
    if d is None:
        d = lambda a, b: float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))

    candidates = R(L(x))
    if not candidates:
        return float("inf")
    return min(d(c, x) for c in candidates)


def adjunction_failure_score(L: ForwardMap, R: ReverseMap,
                              sample: list,
                              d: Callable | None = None) -> dict:
    """Summarise adjunction failure over a sample of points.

    Returns a dict with keys:
      ``mean_error``   mean reconstruction error over the sample
      ``max_error``    worst-case reconstruction error
      ``failure_rate`` fraction of points with error > tol
      ``tol``          tolerance used (Euclidean, 1e-8)
    """
    tol = 1e-8
    errors = [reconstruction_error(L, R, x, d) for x in sample]
    errors_arr = np.array(errors, dtype=float)
    return {
        "mean_error": float(np.mean(errors_arr)),
        "max_error": float(np.max(errors_arr)),
        "failure_rate": float(np.mean(errors_arr > tol)),
        "tol": tol,
    }
