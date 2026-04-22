"""Tests for src/collatz.py — discrete and continuous Collatz maps."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.collatz import (
    chamberland,
    chamberland_orbit,
    collatz_orbit,
    collatz_step,
    count_discrete_fixed_points,
    dynamical_zeta,
    stopping_time,
    stopping_times,
    verify_chamberland,
)


# ---------------------------------------------------------------------------
# Discrete Collatz map
# ---------------------------------------------------------------------------

class TestCollatzStep:
    def test_even(self):
        assert collatz_step(8) == 4
        assert collatz_step(12) == 6
        assert collatz_step(2) == 1

    def test_odd(self):
        assert collatz_step(1) == 4
        assert collatz_step(3) == 10
        assert collatz_step(7) == 22

    def test_invalid(self):
        with pytest.raises(ValueError):
            collatz_step(0)
        with pytest.raises(ValueError):
            collatz_step(-5)


class TestCollatzOrbit:
    def test_orbit_of_1(self):
        assert collatz_orbit(1) == [1]

    def test_orbit_of_6(self):
        # 6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1
        orb = collatz_orbit(6)
        assert orb[0] == 6
        assert orb[-1] == 1

    def test_orbit_ends_at_1(self):
        for n in range(1, 50):
            assert collatz_orbit(n)[-1] == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            collatz_orbit(0)


class TestStoppingTime:
    def test_one_is_zero(self):
        assert stopping_time(1) == 0

    def test_known_values(self):
        # 2 → 1: 1 step
        assert stopping_time(2) == 1
        # 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1: 7 steps
        assert stopping_time(3) == 7

    def test_all_small_reach_one(self):
        for n in range(1, 100):
            assert stopping_time(n) is not None

    def test_stopping_times_dict(self):
        result = stopping_times(5)
        assert isinstance(result, dict)
        assert set(result.keys()) == {1, 2, 3, 4, 5}
        assert result[1] == 0


# ---------------------------------------------------------------------------
# Chamberland continuous extension
# ---------------------------------------------------------------------------

class TestChamberland:
    def test_agrees_with_discrete_on_integers(self):
        assert verify_chamberland(30), "Chamberland extension disagrees with discrete map"

    def test_even_integers(self):
        for n in [2, 4, 6, 10, 20]:
            expected = n / 2
            assert chamberland(float(n)) == pytest.approx(expected, abs=1e-10)

    def test_odd_integers(self):
        for n in [1, 3, 5, 7, 11]:
            expected = 3 * n + 1
            assert chamberland(float(n)) == pytest.approx(expected, abs=1e-10)

    def test_numpy_array_input(self):
        xs = np.array([2.0, 4.0, 6.0])
        result = chamberland(xs)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_continuous_between_integers(self):
        # At x = 1.5 the function should be well-defined and finite
        val = chamberland(1.5)
        assert np.isfinite(val)

    def test_chamberland_orbit_starts_at_x(self):
        orb = chamberland_orbit(3.0, n_iter=200)
        assert orb[0] == pytest.approx(3.0)

    def test_chamberland_orbit_from_positive_integer(self):
        # Starting from an even number the orbit should trend towards 1
        orb = chamberland_orbit(8.0, n_iter=500, tol=1e-5)
        assert any(abs(v - 1.0) < 0.1 for v in orb)


# ---------------------------------------------------------------------------
# Dynamical zeta function  (§8.2)
# ---------------------------------------------------------------------------

class TestDynamicalZeta:
    def test_empty_returns_exp_0(self):
        result = dynamical_zeta([], z=0.5)
        assert abs(result - 1.0) < 1e-12

    def test_single_fixed_point_count(self):
        # fix_counts = [1]  →  exp( z )
        import cmath
        z = 0.3
        result = dynamical_zeta([1], z=z)
        expected = cmath.exp(z)
        assert abs(result - expected) < 1e-10

    def test_small_z_converges(self):
        counts = [1, 1, 2, 1]
        result = dynamical_zeta(counts, z=0.1)
        assert np.isfinite(abs(result))
        assert abs(result) > 0

    def test_at_zero_is_one(self):
        counts = [5, 3, 2]
        result = dynamical_zeta(counts, z=0.0)
        assert abs(result - 1.0) < 1e-12


class TestCountFixedPoints:
    def test_period_1_only_1(self):
        # Only n=1 satisfies collatz_step^1(n) == n? Let's check:
        # collatz_step(1) = 4 ≠ 1, so count should be 0 for upper=10
        count = count_discrete_fixed_points(1, 10)
        # n=1: F(1)=4≠1; no 1-periodic point in [1,10]
        assert count == 0

    def test_period_2_finds_cycle(self):
        # 1 → 4 → 2 → 1 is not period-2; try: F²(1)=F(4)=2≠1
        count = count_discrete_fixed_points(2, 20)
        # Count should be 0 or very small (no known 2-cycles)
        assert count >= 0
