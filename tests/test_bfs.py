"""Tests for src/bfs.py — Bounded Formal System framework."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.bfs import (
    BFS,
    basin_of_attraction,
    boundary_wall,
    decidable_core,
    detect_cycle,
    find_fixed_points,
    in_basin,
    is_fixed_point,
    orbit,
    shadow_region,
)


# ---------------------------------------------------------------------------
# Helpers — simple test systems
# ---------------------------------------------------------------------------

def make_contraction(alpha: float = 0.5) -> BFS:
    """F(x) = alpha * x  —  fixed point at 0, basin = all of ℝ."""
    return BFS(F=lambda x: alpha * x)


def make_shift_left() -> BFS:
    """F(x) = x − 1  —  no fixed point (drifts to −∞)."""
    return BFS(F=lambda x: x - 1.0)


def make_two_attractor() -> BFS:
    """F(x) = sign(x) * |x|^0.9 clipped.

    Fixed point at 0; pushes negative values towards -inf and positive
    values towards 0.  Not the most physical but good for testing basins.
    Using instead: F maps [−5,−0.5] → −1, [0.5,5] → +1 as step function.
    We use a simpler system: the logistic-like  F(x) = 1.5*x*(1 − x)  on [0,1]
    with a fixed point at x* = 1/3.
    """
    return BFS(F=lambda x: 1.5 * x * (1.0 - x))


# ---------------------------------------------------------------------------
# Tests: BFS construction
# ---------------------------------------------------------------------------

class TestBFSConstruction:
    def test_custom_metric(self):
        custom_d = lambda a, b: abs(a - b) ** 2
        bfs = BFS(F=lambda x: x, d=custom_d)
        assert bfs.d(0.0, 2.0) == pytest.approx(4.0)

    def test_default_euclidean_scalar(self):
        bfs = BFS(F=lambda x: x)
        assert bfs.d(3.0, 7.0) == pytest.approx(4.0)

    def test_default_euclidean_vector(self):
        bfs = BFS(F=lambda x: x)
        assert bfs.d(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests: Fixed points (Definition 2)
# ---------------------------------------------------------------------------

class TestFixedPoints:
    def test_identity_all_fixed(self):
        bfs = BFS(F=lambda x: x)
        pts = [1.0, 2.0, 3.0]
        fps = find_fixed_points(bfs, pts)
        assert fps == pts

    def test_contraction_fixed_at_zero(self):
        bfs = make_contraction(0.5)
        fps = find_fixed_points(bfs, [0.0, 1.0, -1.0])
        assert 0.0 in fps
        assert 1.0 not in fps

    def test_not_fixed(self):
        bfs = BFS(F=lambda x: x + 1.0)
        assert not is_fixed_point(bfs, 5.0)

    def test_fixed_at_nonzero(self):
        # F(x) = 2 − x  →  fixed point at x = 1
        bfs = BFS(F=lambda x: 2.0 - x)
        assert is_fixed_point(bfs, 1.0)
        assert not is_fixed_point(bfs, 0.0)


# ---------------------------------------------------------------------------
# Tests: orbit (Definition 2)
# ---------------------------------------------------------------------------

class TestOrbit:
    def test_orbit_length(self):
        bfs = BFS(F=lambda x: x * 0.5)
        o = orbit(bfs, 8.0, 3)
        assert len(o) == 4  # x, F(x), F²(x), F³(x)

    def test_orbit_values(self):
        bfs = BFS(F=lambda x: x * 0.5)
        o = orbit(bfs, 8.0, 3)
        assert o == pytest.approx([8.0, 4.0, 2.0, 1.0])

    def test_orbit_fixed_stays(self):
        bfs = BFS(F=lambda x: x)
        o = orbit(bfs, 42.0, 5)
        assert all(v == pytest.approx(42.0) for v in o)


# ---------------------------------------------------------------------------
# Tests: detect_cycle (Definition 2)
# ---------------------------------------------------------------------------

class TestDetectCycle:
    def test_fixed_point_is_period_one_cycle(self):
        # F(x) = 2 − x  fixed point x=1, cycle = [1]
        bfs = BFS(F=lambda x: 2.0 - x)
        cyc = detect_cycle(bfs, 1.0)
        assert cyc is not None
        assert any(abs(v - 1.0) < 1e-6 for v in cyc)

    def test_two_cycle(self):
        # F(x) = 1 − x  has 2-cycle: 0 → 1 → 0
        bfs = BFS(F=lambda x: 1.0 - x)
        cyc = detect_cycle(bfs, 0.0)
        assert cyc is not None
        assert len(cyc) in (1, 2)  # Floyd may land on either point


# ---------------------------------------------------------------------------
# Tests: Basin of attraction (Definition 3)
# ---------------------------------------------------------------------------

class TestBasin:
    def test_full_basin_contraction(self):
        bfs = make_contraction(0.5)
        candidates = np.linspace(-5.0, 5.0, 51).tolist()
        b = basin_of_attraction(bfs, 0.0, candidates)
        assert len(b) == len(candidates)

    def test_empty_basin_no_attractor(self):
        # F(x) = x + 1 never reaches 0
        bfs = BFS(F=lambda x: x + 1.0)
        b = basin_of_attraction(bfs, 0.0, [1.0, 2.0, 3.0], n_iter=20)
        assert b == []

    def test_in_basin_true(self):
        bfs = make_contraction(0.5)
        assert in_basin(bfs, 100.0, 0.0, n_iter=10_000)

    def test_in_basin_false(self):
        bfs = BFS(F=lambda x: x + 1.0)
        assert not in_basin(bfs, 5.0, 0.0, n_iter=50)


# ---------------------------------------------------------------------------
# Tests: Decidable core D(S) and Shadow U(S) (Definitions 4 & 5)
# ---------------------------------------------------------------------------

class TestDecidableCoreShadow:
    def test_all_in_core_for_contraction(self):
        bfs = make_contraction(0.5)
        candidates = np.linspace(-3.0, 3.0, 31).tolist()
        core = decidable_core(bfs, candidates, [0.0])
        assert len(core) == len(candidates)

    def test_shadow_empty_when_all_converge(self):
        bfs = make_contraction(0.5)
        candidates = np.linspace(-3.0, 3.0, 31).tolist()
        core = decidable_core(bfs, candidates, [0.0])
        shadow = shadow_region(candidates, core)
        assert shadow == []

    def test_shadow_nonempty_when_some_diverge(self):
        # F(x) = 2*x: contracts only at 0; large values diverge
        bfs = BFS(F=lambda x: 2.0 * x)
        candidates = [0.0, 0.5, 1.0, 10.0]
        core = decidable_core(bfs, candidates, [0.0], n_iter=200)
        shadow = shadow_region(candidates, core)
        # 0 is in core (it's the fixed point); others diverge
        assert 0.0 in core
        assert len(shadow) > 0


# ---------------------------------------------------------------------------
# Tests: Boundary wall ∂S (Definition 6)
# ---------------------------------------------------------------------------

class TestBoundaryWall:
    def test_boundary_between_core_and_shadow(self):
        # Simple 1-D case: core = [0..4], shadow = [6..10]
        candidates = np.arange(0, 11, dtype=float)
        core_mask = candidates <= 4.5
        bnd = boundary_wall(candidates, core_mask, epsilon=1.5)
        # Points 4 and 5 should be on the boundary
        assert bnd[4] or bnd[5]

    def test_all_same_label_no_boundary(self):
        candidates = np.linspace(0.0, 1.0, 20)
        core_mask = np.ones(20, dtype=bool)
        bnd = boundary_wall(candidates, core_mask, epsilon=0.1)
        assert not bnd.any()
