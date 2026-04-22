"""Polynomial map demo — fixed points, basins, shadow region, entropy.

Run:
    python examples/polynomial_demo.py

Demonstrates the BFS framework using the logistic map  F(x) = r·x·(1−x)
on the interval [0, 1], which transitions from simple (r < 3) to chaotic
(r = 4) as r increases.

Produces:
  • examples/logistic_basins.png    — basin / shadow / boundary for r = 3.5
  • examples/logistic_bifurcation.png — bifurcation diagram

Also demonstrates the adjunction module: forward = F, reverse = approximate
pre-image via Newton iteration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from src.bfs import (
    BFS,
    decidable_core,
    shadow_region,
    boundary_wall,
    orbit,
    find_fixed_points,
)
from src.entropy import orbit_complexity, lyapunov_exponent
from src.adjunction import ForwardMap, ReverseMap, adjunction_failure_score


# ---------------------------------------------------------------------------
# Logistic map helpers
# ---------------------------------------------------------------------------

def logistic(r: float):
    """Return the logistic map  F(x) = r·x·(1−x)."""
    return lambda x: r * x * (1.0 - x)


def logistic_fixed_points_analytic(r: float) -> list[float]:
    """Exact fixed points of  F(x) = r·x·(1−x): 0 and (r−1)/r."""
    if r <= 1.0:
        return [0.0]
    return [0.0, (r - 1.0) / r]


# ---------------------------------------------------------------------------
# Basin / shadow / boundary plot
# ---------------------------------------------------------------------------

def plot_basins(r: float = 3.5, n_pts: int = 500,
                output_path: str | None = None) -> None:
    grid = np.linspace(0.01, 0.99, n_pts)
    # For r = 3.5 the fixed point (r-1)/r ≈ 0.714 is unstable; the map has a
    # 4-cycle.  We treat the interval [0.3, 0.9] as the approximate attractor
    # region (D(S)) and everything else as the shadow.
    bfs = BFS(F=logistic(r))

    # Use decidable_core with multiple attractor candidates spanning the 4-cycle
    # Approximate 4-cycle points for r=3.5
    x = 0.5
    for _ in range(1000):
        x = logistic(r)(x)
    attractor_pts = [x]
    for _ in range(3):
        x = logistic(r)(x)
        attractor_pts.append(x)

    core = decidable_core(bfs, grid.tolist(), attractor_pts, n_iter=2000, tol=0.05)
    shadow = shadow_region(grid.tolist(), core)

    core_arr = np.array(core)
    shadow_arr = np.array(shadow) if shadow else np.array([])

    # Build mask and boundary
    core_mask = np.array([any(np.isclose(float(x), float(c)) for c in core)
                          for x in grid.tolist()])
    bnd = boundary_wall(grid, core_mask, epsilon=4.0 / n_pts)

    fig, ax = plt.subplots(figsize=(10, 3))
    if core_arr.size:
        ax.scatter(core_arr, np.zeros_like(core_arr),
                   c="steelblue", s=4, label=f"D(S) ({len(core_arr)})")
    if shadow_arr.size:
        ax.scatter(shadow_arr, np.zeros_like(shadow_arr),
                   c="tomato", s=4, label=f"U(S) ({len(shadow_arr)})")
    boundary_pts = grid[bnd]
    if boundary_pts.size:
        ax.scatter(boundary_pts, np.zeros_like(boundary_pts),
                   c="gold", s=15, zorder=5, label=f"∂S ({boundary_pts.size})")

    ax.set_title(f"Logistic map r={r}: decidable core D(S), shadow U(S), boundary ∂S")
    ax.set_xlabel("x")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_yticks([])
    plt.tight_layout()

    path = output_path or os.path.join(os.path.dirname(__file__), "logistic_basins.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Bifurcation diagram
# ---------------------------------------------------------------------------

def plot_bifurcation(output_path: str | None = None) -> None:
    r_vals = np.linspace(2.5, 4.0, 800)
    n_warmup, n_plot = 500, 200
    xs, rs = [], []
    for r in r_vals:
        x = 0.5
        for _ in range(n_warmup):
            x = r * x * (1.0 - x)
        for _ in range(n_plot):
            x = r * x * (1.0 - x)
            xs.append(x)
            rs.append(r)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(rs, xs, s=0.2, alpha=0.3, color="black")
    ax.set_xlabel("r")
    ax.set_ylabel("x (attractor)")
    ax.set_title("Logistic map bifurcation diagram")
    plt.tight_layout()

    path = output_path or os.path.join(os.path.dirname(__file__), "logistic_bifurcation.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Adjunction failure demo
# ---------------------------------------------------------------------------

def demo_adjunction(r: float = 3.9) -> None:
    """Show that the logistic forward map is non-injective (2-to-1).

    Pre-image of y = F(x) = r·x·(1−x) is obtained by solving the quadratic:
    r·x² − r·x + y = 0  →  x = (1 ± √(1 − 4y/r)) / 2
    """
    def L_apply(x: float) -> float:
        return r * x * (1.0 - x)

    def R_apply(y: float) -> list[float]:
        discriminant = 1.0 - 4.0 * y / r
        if discriminant < 0:
            return []
        sqrt_d = np.sqrt(discriminant)
        return [(1.0 - sqrt_d) / 2.0, (1.0 + sqrt_d) / 2.0]

    L = ForwardMap(apply=L_apply, is_injective=False)
    R = ReverseMap(apply=R_apply, is_exact=True)

    sample = np.linspace(0.01, 0.99, 50).tolist()
    score = adjunction_failure_score(L, R, sample)

    print(f"\n── Adjunction failure demo (logistic r={r}) ──")
    print(f"  R is exact:      {R.is_exact}")
    print(f"  Mean recon. err: {score['mean_error']:.2e}")
    print(f"  Max  recon. err: {score['max_error']:.2e}")
    print(f"  Failure rate:    {score['failure_rate']:.1%}  (err > {score['tol']:.0e})")
    print("  Note: R gives 2 pre-images (2-to-1 map); exact round-trip is")
    print("  possible only when we pick the *correct* branch (C2 complementarity).")


# ---------------------------------------------------------------------------
# Entropy demo
# ---------------------------------------------------------------------------

def demo_entropy() -> None:
    print("\n── Topological entropy / Lyapunov exponents ──")
    for r in [2.0, 3.2, 3.5, 4.0]:
        F = logistic(r)
        lam = lyapunov_exponent(F, x0=0.3, n_iter=5000)
        grid = np.linspace(0.01, 0.99, 100)
        h_approx = orbit_complexity(F, grid, n_iter=8, epsilon=0.05)
        print(f"  r={r:.1f}  λ≈{lam:+.3f}  h_approx≈{h_approx:.3f}"
              f"  ({'positive entropy' if lam > 0 else 'contracting'})")


if __name__ == "__main__":
    demo_entropy()
    demo_adjunction(r=3.9)
    plot_basins(r=3.5)
    plot_bifurcation()
    print("\nDone.")
