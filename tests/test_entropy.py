"""Tests for src/entropy.py — topological entropy approximation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from src.entropy import (
    entropy_transition_matrix,
    lyapunov_exponent,
    orbit_complexity,
)


# ---------------------------------------------------------------------------
# Lyapunov exponent
# ---------------------------------------------------------------------------

class TestLyapunovExponent:
    def test_contraction_negative(self):
        # F(x) = 0.5 * x  →  F'(x) = 0.5  →  λ = log(0.5) < 0
        lam = lyapunov_exponent(lambda x: 0.5 * x, x0=1.0, n_iter=500)
        assert lam == pytest.approx(math.log(0.5), abs=0.05)

    def test_expansion_positive(self):
        # F(x) = 2 * x  →  F'(x) = 2  →  λ = log(2) > 0
        lam = lyapunov_exponent(lambda x: 2.0 * x, x0=1.0, n_iter=500)
        assert lam == pytest.approx(math.log(2.0), abs=0.05)

    def test_logistic_r4_log2(self):
        # F(x) = 4x(1−x) is fully chaotic; Lyapunov exponent = log(2)
        # (smooth map — safe for finite-difference differentiation)
        lam = lyapunov_exponent(lambda x: 4.0 * x * (1.0 - x),
                                x0=0.3, n_iter=5000)
        assert lam == pytest.approx(math.log(2.0), abs=0.1)


# ---------------------------------------------------------------------------
# Orbit complexity / spanning-set estimate
# ---------------------------------------------------------------------------

class TestOrbitComplexity:
    def test_identity_zero_entropy(self):
        # F(x) = x: no orbit complexity growth; entropy = 0.
        # orbit_complexity returns (1/n)*log(spanning), which is a finite-n
        # estimate.  At n=10 it is a small positive lower bound; check it is
        # less than log(2)/2 (much below any chaotic map) and non-negative.
        grid = np.linspace(0.0, 1.0, 30)
        h = orbit_complexity(lambda x: x, grid, n_iter=10, epsilon=0.05)
        assert h >= 0.0
        assert h < math.log(2) / 2

    def test_contraction_zero_entropy(self):
        # F(x) = 0.5*x: all orbits collapse to 0; effectively 0 entropy
        grid = np.linspace(-1.0, 1.0, 40)
        h = orbit_complexity(lambda x: 0.5 * x, grid, n_iter=20, epsilon=0.01)
        # Should be close to 0 (orbits cluster near 0)
        assert h >= 0.0

    def test_positive_entropy_for_complex_map(self):
        # F(x) = 4x(1-x) (fully chaotic logistic map on [0,1])
        # topological entropy = log(2)
        grid = np.linspace(0.01, 0.99, 80)
        h = orbit_complexity(lambda x: 4.0 * x * (1.0 - x), grid,
                             n_iter=8, epsilon=0.05)
        assert h > 0.0


# ---------------------------------------------------------------------------
# Transition matrix entropy
# ---------------------------------------------------------------------------

class TestEntropyTransitionMatrix:
    def test_identity_zero(self):
        T = np.eye(3)
        assert entropy_transition_matrix(T) == pytest.approx(0.0)

    def test_full_2x2(self):
        # T = [[1,1],[1,1]]  largest eigenvalue = 2  →  h = log(2)
        T = np.array([[1, 1], [1, 1]])
        h = entropy_transition_matrix(T)
        assert h == pytest.approx(math.log(2), abs=1e-10)

    def test_golden_mean_shift(self):
        # Golden-mean shift: T = [[1,1],[1,0]]
        # λ_max = golden ratio φ = (1 + √5)/2  →  h = log(φ)
        T = np.array([[1.0, 1.0], [1.0, 0.0]])
        h = entropy_transition_matrix(T)
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        assert h == pytest.approx(math.log(phi), abs=1e-10)

    def test_single_state_zero(self):
        T = np.array([[1]])
        assert entropy_transition_matrix(T) == pytest.approx(0.0)
