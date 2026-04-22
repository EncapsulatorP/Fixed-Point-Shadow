"""Topological entropy approximation.

Implements Definition 8 from the README:

  h(F) measures exponential growth in distinguishable orbit segments.

The standard finite-resolution approximation of topological entropy uses
*spanning sets* (Adler–Konheim–McAndrew 1965): a set E is (n, ε)-spanning
for a compact metric space (X, d) if for every x ∈ X there is a y ∈ E
with  max_{0≤k≤n-1} d(Fᵏ(x), Fᵏ(y)) < ε.

h(F, ε) = lim_{n→∞} (1/n) log r(n, ε)

where r(n, ε) is the minimum cardinality of an (n, ε)-spanning set.
As ε → 0 this recovers the true topological entropy.

For a one-dimensional map on an interval we also provide a simpler
*slope-based* lower-bound estimate via the mean expansion rate.

References
----------
Adler, R., Konheim, A., McAndrew, M. (1965). Topological entropy.
    Trans. Amer. Math. Soc. 114, 309-319.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Orbit-complexity / spanning-set estimate
# ---------------------------------------------------------------------------

def orbit_complexity(F, grid: np.ndarray, n_iter: int, epsilon: float) -> float:
    """Estimate topological entropy via spanning-set growth.

    Algorithm
    ---------
    1. Iterate F on every point in *grid* for *n_iter* steps, recording the
       full orbit segment of length *n_iter*.
    2. Greedily count the number of orbit segments that are > ε apart in the
       sup-norm (dynamical metric  dₙ(x,y) = max_{k<n} d(Fᵏ(x),Fᵏ(y))).
    3. Return  (1/n_iter) · log( count ).

    Parameters
    ----------
    F:
        Map  F : ℝ → ℝ  (scalar callable).
    grid:
        1-D array of sample points.
    n_iter:
        Orbit segment length n.
    epsilon:
        Resolution ε for distinguishing orbits.

    Returns
    -------
    Non-negative float: estimated h(F, ε, n).

    Notes
    -----
    This is a *lower bound* on the true topological entropy; it depends on
    the density of *grid* and grows towards h(F) as grid → dense and ε → 0.
    """
    grid = np.asarray(grid, dtype=float).ravel()
    n_pts = len(grid)

    # Build orbit matrix: rows = points, columns = iterate index
    orbits = np.empty((n_pts, n_iter), dtype=float)
    x = grid.copy()
    for k in range(n_iter):
        orbits[:, k] = x
        x = np.vectorize(F)(x)

    # Greedy spanning-set count
    spanning = 0
    used = np.zeros(n_pts, dtype=bool)

    for i in range(n_pts):
        if used[i]:
            continue
        spanning += 1
        # Mark all points within epsilon in the sup-norm (dynamical metric)
        dyn_dist = np.max(np.abs(orbits - orbits[i]), axis=1)
        used |= dyn_dist < epsilon

    if spanning <= 1:
        return 0.0
    return math.log(spanning) / n_iter


# ---------------------------------------------------------------------------
# Lyapunov / slope-based estimate (1-D maps)
# ---------------------------------------------------------------------------

def lyapunov_exponent(F, x0: float, n_iter: int = 10_000,
                      delta: float = 1e-7) -> float:
    """Estimate the Lyapunov exponent via finite-difference derivative.

    λ = lim_{n→∞} (1/n) Σ_{k=0}^{n-1} log |F'(Fᵏ(x₀))|

    The derivative is approximated numerically:
    F'(x) ≈ (F(x + δ) − F(x − δ)) / (2δ)

    A positive λ is the standard signature of chaotic / complex dynamics.
    It provides a lower bound on topological entropy for smooth maps on
    intervals:  h(F) ≥ max(0, λ).

    Parameters
    ----------
    F:
        Scalar map F : ℝ → ℝ.
    x0:
        Starting point for the orbit.
    n_iter:
        Number of iterates to average over.
    delta:
        Step size for numerical differentiation.

    Returns
    -------
    Estimated Lyapunov exponent (can be negative for contracting dynamics).
    """
    x = float(x0)
    total = 0.0
    valid = 0
    for _ in range(n_iter):
        deriv = (F(x + delta) - F(x - delta)) / (2.0 * delta)
        if deriv != 0.0:
            total += math.log(abs(deriv))
            valid += 1
        x = F(x)
    return total / valid if valid > 0 else 0.0


# ---------------------------------------------------------------------------
# Entropy for symbolic / finite-state maps
# ---------------------------------------------------------------------------

def entropy_transition_matrix(T: np.ndarray) -> float:
    """Compute topological entropy from a transition matrix.

    For a subshift of finite type defined by a 0/1 transition matrix *T*,
    the topological entropy equals log(λ_max) where λ_max is the largest
    real eigenvalue of T.

    Parameters
    ----------
    T:
        Square (0/1) numpy array.

    Returns
    -------
    Topological entropy  h = log(λ_max).

    References
    ----------
    Adler–Konheim–McAndrew (1965).
    """
    eigenvalues = np.linalg.eigvals(T.astype(float))
    lambda_max = float(np.max(np.abs(eigenvalues)))
    return math.log(lambda_max) if lambda_max > 1.0 else 0.0
