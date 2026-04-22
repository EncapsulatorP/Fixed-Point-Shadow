"""Collatz demo — discrete and continuous (Chamberland) maps.

Run:
    python examples/collatz_demo.py

Produces two plots:
  1. Stopping-time histogram for n ∈ [1, 1000].
  2. Chamberland orbits from several starting real values.

Saves figures to examples/collatz_stopping_times.png and
examples/chamberland_orbits.png.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs

from src.collatz import (
    stopping_times,
    chamberland,
    chamberland_orbit,
    verify_chamberland,
)
from src.bfs import BFS, decidable_core, shadow_region, boundary_wall


def plot_stopping_times(upper: int = 1000, output_path: str | None = None) -> None:
    """Plot stopping-time histogram and scatter for n ∈ [1, upper]."""
    st = stopping_times(upper)
    ns = list(st.keys())
    times = [st[n] for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(ns, times, s=1, alpha=0.4, color="steelblue")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Stopping time")
    axes[0].set_title(f"Collatz stopping times (n ≤ {upper})")

    axes[1].hist(times, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("Stopping time")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of stopping times")

    plt.tight_layout()
    path = output_path or os.path.join(os.path.dirname(__file__), "collatz_stopping_times.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_chamberland_orbits(starts: list[float] | None = None,
                            output_path: str | None = None) -> None:
    """Plot Chamberland orbits for several real starting values."""
    if starts is None:
        starts = [2.5, 3.7, 5.1, 7.0, 11.3]

    fig, ax = plt.subplots(figsize=(10, 5))
    for x0 in starts:
        orb = chamberland_orbit(x0, n_iter=300, tol=1e-5)
        ax.plot(orb, label=f"x₀ = {x0}", linewidth=1.2)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="y = 1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Chamberland continuous-extension orbits")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = output_path or os.path.join(os.path.dirname(__file__), "chamberland_orbits.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def demo_bfs_collatz(upper: int = 50) -> None:
    """Show D(S) and U(S) for the BFS built on the discrete Collatz map."""
    from src.collatz import collatz_step

    bfs = BFS(F=collatz_step, d=lambda a, b: abs(a - b))
    candidates = list(range(1, upper + 1))
    # Attractor: the cycle {1, 2, 4} under the standard Collatz map
    # The cycle visits 1 → 4 → 2 → 1, so we treat 1 as the attractor
    core = decidable_core(bfs, candidates, attractors=[1], n_iter=1000, tol=0.5)
    shadow = shadow_region(candidates, core)

    print(f"\n── Discrete Collatz BFS (n ∈ [1, {upper}]) ──")
    print(f"  D(S) size : {len(core)}   (converge to 1)")
    print(f"  U(S) size : {len(shadow)} (did not converge within budget)")
    if shadow:
        print(f"  U(S) sample: {shadow[:10]}")


if __name__ == "__main__":
    print("Verifying Chamberland extension agrees with discrete Collatz on integers:")
    ok = verify_chamberland(30)
    print(f"  ✓ Verified for n ∈ [1, 30]" if ok else "  ✗ MISMATCH detected")

    demo_bfs_collatz(upper=200)
    plot_stopping_times(upper=1000)
    plot_chamberland_orbits()
    print("\nDone.")
