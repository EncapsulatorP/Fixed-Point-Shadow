"""Collatz map — discrete and continuous implementations.

Covers §7.2 and §8.3 of the README:

  * Standard (discrete) Collatz map on positive integers.
  * Chamberland (1996) continuous extension to ℝ.
  * Orbit computation and stopping-time statistics.
  * Dynamical zeta function  ζ_F(z) = exp(Σ |Fix(Fⁿ)|/n · zⁿ)  (§8.2),
    evaluated for the discrete Collatz map on a finite initial segment.

References
----------
Chamberland, M. (1996). *A continuous extension of the 3x+1 problem
    to the real line.*  Dynam. Contin. Discrete Impuls. Systems 2(4), 495-509.
Tao, T. (2019). *Almost all orbits of the Collatz map attain almost
    bounded values.*  Forum Math. Pi 10 (2022), e29.
"""

from __future__ import annotations

import math
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# Discrete Collatz map
# ---------------------------------------------------------------------------

def collatz_step(n: int) -> int:
    """Apply one step of the standard Collatz map.

    F(n) = n/2       if n is even
    F(n) = 3n + 1    if n is odd

    Parameters
    ----------
    n:
        A positive integer.

    Raises
    ------
    ValueError:
        If n < 1.
    """
    if n < 1:
        raise ValueError(f"collatz_step requires a positive integer, got {n}")
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_orbit(n: int, max_iter: int = 10_000) -> list[int]:
    """Return the Collatz orbit  [n, F(n), F²(n), …, 1].

    Terminates when the orbit reaches 1 or after *max_iter* steps.

    Parameters
    ----------
    n:
        Starting positive integer.
    max_iter:
        Safety cap on the number of iterations.

    Returns
    -------
    List of orbit values ending at 1 (if reached within *max_iter* steps).
    """
    if n < 1:
        raise ValueError(f"collatz_orbit requires a positive integer, got {n}")
    orb = [n]
    for _ in range(max_iter):
        if n == 1:
            break
        n = collatz_step(n)
        orb.append(n)
    return orb


def stopping_time(n: int, max_iter: int = 10_000) -> int | None:
    """Return the number of steps for the orbit of n to reach 1.

    Returns *None* if 1 is not reached within *max_iter* steps.
    """
    if n < 1:
        raise ValueError(f"stopping_time requires a positive integer, got {n}")
    if n == 1:
        return 0
    steps = 0
    cur = n
    for _ in range(max_iter):
        cur = collatz_step(cur)
        steps += 1
        if cur == 1:
            return steps
    return None


def stopping_times(upper: int, max_iter: int = 10_000) -> dict[int, int | None]:
    """Compute stopping times for all integers in [1, upper]."""
    return {n: stopping_time(n, max_iter) for n in range(1, upper + 1)}


# ---------------------------------------------------------------------------
# Continuous extension (Chamberland 1996)
# ---------------------------------------------------------------------------

def chamberland(x: float | np.ndarray) -> float | np.ndarray:
    """Chamberland's continuous extension of the Collatz map to ℝ.

    F(x) = (1/4)(2 + 7x − (2 + 5x) cos(πx))

    Verification on integers
    ────────────────────────
    Even n:  cos(πn) = +1  →  F(n) = (1/4)(2 + 7n − 2 − 5n) = n/2   ✓
    Odd  n:  cos(πn) = −1  →  F(n) = (1/4)(2 + 7n + 2 + 5n) = 3n+1  ✓

    Parameters
    ----------
    x:
        A real number or numpy array of real numbers.

    References
    ----------
    Chamberland (1996), equation (1.2).
    """
    x = np.asarray(x, dtype=float)
    return 0.25 * (2.0 + 7.0 * x - (2.0 + 5.0 * x) * np.cos(np.pi * x))


def chamberland_orbit(x: float, n_iter: int = 200, tol: float = 1e-6) -> list[float]:
    """Return the Chamberland orbit  [x, F(x), F²(x), …].

    Stops early if the orbit reaches a value close to 1.0 or diverges
    beyond a threshold.

    Parameters
    ----------
    x:
        Real starting value.
    n_iter:
        Maximum iterations.
    tol:
        Convergence tolerance (orbit declared converged when |Fᵏ(x)−1|<tol).
    """
    pts = [x]
    cur = float(x)
    for _ in range(n_iter):
        if abs(cur - 1.0) < tol:
            break
        if abs(cur) > 1e12:
            break
        cur = float(chamberland(cur))
        pts.append(cur)
    return pts


def chamberland_generator(x: float) -> Generator[float, None, None]:
    """Infinite generator of Chamberland iterates; caller controls termination."""
    cur = float(x)
    while True:
        yield cur
        cur = float(chamberland(cur))


# ---------------------------------------------------------------------------
# Collatz-related dynamical zeta function  (§8.2 – §8.3)
# ---------------------------------------------------------------------------

def count_discrete_fixed_points(period: int, upper: int) -> int:
    """Count integers in [1, upper] that are periodic under the Collatz map
    with period dividing *period*.

    For the standard Collatz map the only known cycle is {1, 2} (period 2
    under the convention F(1) = 4, F(2) = 1 — reachable from 1 → 4 → 2 → 1
    — or equivalently the cycle 1 → 4 → 2 → 1 has period 3 under the
    standard map).  This function counts by direct iteration up to *upper*.

    Notes
    -----
    This is a *finite approximation* of |Fix(Fⁿ)| restricted to [1, upper].
    It is used only to illustrate the dynamical zeta construction; the full
    count over all of ℤ₊ is unknown for most periods > 1.
    """
    count = 0
    for n in range(1, upper + 1):
        cur = n
        found = False
        for _ in range(period):
            cur = collatz_step(cur)
            if cur > upper * 10:
                break
        if cur == n:
            found = True
        count += int(found)
    return count


def dynamical_zeta(fix_counts: list[int], z: complex) -> complex:
    """Evaluate the dynamical zeta function at *z*.

    ζ_F(z) = exp( Σ_{n≥1}  |Fix(Fⁿ)| / n  ·  zⁿ )

    Parameters
    ----------
    fix_counts:
        List [|Fix(F¹)|, |Fix(F²)|, …, |Fix(Fᴺ)|].
    z:
        Complex evaluation point.

    Returns
    -------
    Complex value of ζ_F(z).

    Notes
    -----
    The series is truncated at len(fix_counts) terms.  Convergence is only
    guaranteed for |z| small enough; this is an *approximation*.
    See §8.2 of the README.
    """
    s = sum(cnt / (k + 1) * z ** (k + 1)
            for k, cnt in enumerate(fix_counts))
    return complex(np.exp(s))


# ---------------------------------------------------------------------------
# Utility: verify Chamberland agrees with the discrete map on integers
# ---------------------------------------------------------------------------

def verify_chamberland(n_max: int = 20) -> bool:
    """Check that chamberland(n) == collatz_step(n) for n in [1, n_max]."""
    for n in range(1, n_max + 1):
        expected = collatz_step(n)
        got = round(chamberland(float(n)))
        if got != expected:
            return False
    return True
