"""
Visual test for track winding canonicalization.

Requires the JS API server to be running on localhost:4242.
Run:  python -m mapelite.tests.test_canonicalization_visual
      (or just: python mapelite/tests/test_canonicalization_visual.py)

Produces two PNG files in the same directory:
  - canon_mirror_test.png   — mirrored tracks must converge to the same shape
  - canon_direction_test.png — reversed (forward/backward) tracks must remain distinct
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import requests

BASE_URL = "http://localhost:4242"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_ORIG   = "#2166ac"
COLOR_MOD    = "#d6604d"
COLOR_CANON1 = "#4dac26"
COLOR_CANON2 = "#f1a340"


# ─── helpers ────────────────────────────────────────────────────────────────

def signed_area(spline):
    x = np.array([p["x"] for p in spline])
    y = np.array([p["y"] for p in spline])
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def spline_to_xy(spline):
    return (
        np.array([p["x"] for p in spline]),
        np.array([p["y"] for p in spline]),
    )


def mirror_track(spline):
    """Flip winding order by mirroring x around the track centre."""
    xs = [p["x"] for p in spline]
    cx = (min(xs) + max(xs)) / 2
    return [{"x": 2 * cx - p["x"], "y": p["y"]} for p in spline]


def reverse_track(spline):
    """Reverse the traversal direction (forward ↔ backward)."""
    return list(reversed(spline))


def generate_track(seed: int, track_size: int = 6) -> list:
    r = requests.post(
        f"{BASE_URL}/generate",
        json={"id": seed, "mode": "voronoi", "trackSize": track_size, "rngMode": 0},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["splineVector"]


def canonicalize(spline: list) -> list:
    r = requests.post(
        f"{BASE_URL}/canonicalize",
        json={"splineVector": spline},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["splineVector"]


# ─── plotting ────────────────────────────────────────────────────────────────

def _draw_track(ax, spline, color, title, lw=1.4):
    x, y = spline_to_xy(spline)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round")
    ax.plot(x[0], y[0], "o", color=color, markersize=5, zorder=5)

    # Place arrowheads at evenly-spaced points, oriented along the local tangent.
    # The shaft length (eps) is kept tiny so only the arrowhead triangle is visible.
    scale = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0) * 0.006
    n_arrows = 6
    step = max(len(x) // n_arrows, 1)
    for i in range(step // 2, len(x) - 1, step):
        dx = x[min(i + 1, len(x) - 1)] - x[max(i - 1, 0)]
        dy = y[min(i + 1, len(y) - 1)] - y[max(i - 1, 0)]
        norm = np.hypot(dx, dy)
        if norm < 1e-10:
            continue
        ux, uy = dx / norm * scale, dy / norm * scale
        ax.annotate(
            "",
            xy=(x[i] + ux, y[i] + uy),
            xytext=(x[i] - ux, y[i] - uy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                            mutation_scale=12, fc=color),
            annotation_clip=False,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    area = signed_area(spline)
    winding = "CCW" if area < 0 else "CW"
    ax.set_title(f"{title}\n[{winding}, area={area:.0f}]", pad=4)


def _pad_axes_to_common_limits(axs_row):
    """Force all axes in a row to share the same data extents."""
    xlims = [ax.get_xlim() for ax in axs_row]
    ylims = [ax.get_ylim() for ax in axs_row]
    xl = (min(l[0] for l in xlims), max(l[1] for l in xlims))
    yl = (min(l[0] for l in ylims), max(l[1] for l in ylims))
    for ax in axs_row:
        ax.set_xlim(xl)
        ax.set_ylim(yl)


# ─── test cases ──────────────────────────────────────────────────────────────

SEEDS = [42, 123, 999]


def test_mirror_invariance():
    """
    After canonicalization, a track and its mirror image must be identical.
    Column layout:  Original | Mirrored | Canon(Original) | Canon(Mirrored)
    """
    n = len(SEEDS)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    fig.suptitle(
        "Canonicalization — Mirror Invariance\n"
        "Canon(Original) and Canon(Mirrored) must match",
        fontsize=11,
        y=1.01,
    )

    results = []
    for row, seed in enumerate(SEEDS):
        axs = axes[row]
        original = generate_track(seed)
        mirrored = mirror_track(original)
        c_orig = canonicalize(original)
        c_mirr = canonicalize(mirrored)

        _draw_track(axs[0], original, COLOR_ORIG,  f"Seed {seed} — original")
        _draw_track(axs[1], mirrored, COLOR_MOD,   "Mirrored (flipped)")
        _draw_track(axs[2], c_orig,   COLOR_CANON1, "Canon(Original)")
        _draw_track(axs[3], c_mirr,   COLOR_CANON2, "Canon(Mirrored)")
        _pad_axes_to_common_limits(axs)

        xo, yo = spline_to_xy(c_orig)
        xm, ym = spline_to_xy(c_mirr)
        if len(xo) == len(xm):
            match = np.allclose(xo, xm, atol=1.0) and np.allclose(yo, ym, atol=1.0)
        else:
            match = False
        label = "PASS" if match else "FAIL"
        results.append((seed, match))
        axs[3].set_title(axs[3].get_title() + f"\n[{label}]", pad=4)
        if match:
            for ax in (axs[2], axs[3]):
                ax.patch.set_linewidth(1.5)
                ax.patch.set_edgecolor("#4dac26")
                ax.patch.set_visible(True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "canon_mirror_test.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

    passed = sum(1 for _, ok in results if ok)
    print(f"Mirror invariance: {passed}/{len(SEEDS)} passed")
    for seed, ok in results:
        print(f"  seed={seed:>4}  {'OK' if ok else 'FAIL'}")
    return passed == len(SEEDS)


def test_direction_preserved():
    """
    Reversing traversal direction is NOT eliminated by canonicalization —
    forward and backward versions of the same track must remain distinct.
    Column layout:  Forward | Backward | Canon(Forward) | Canon(Backward)
    """
    n = len(SEEDS)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    fig.suptitle(
        "Canonicalization — Direction Preserved\n"
        "Canon(Forward) and Canon(Backward) should be different shapes",
        fontsize=11,
        y=1.01,
    )

    for row, seed in enumerate(SEEDS):
        axs = axes[row]
        original = generate_track(seed)
        backward = reverse_track(original)
        c_fwd = canonicalize(original)
        c_bwd = canonicalize(backward)

        _draw_track(axs[0], original,  COLOR_ORIG,  f"Seed {seed} — forward (CCW)")
        _draw_track(axs[1], backward,  COLOR_MOD,   "Backward (reversed)")
        _draw_track(axs[2], c_fwd,     COLOR_CANON1, "Canon(Forward)")
        _draw_track(axs[3], c_bwd,     "#762a83",    "Canon(Backward)")
        _pad_axes_to_common_limits(axs)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "canon_direction_test.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to {BASE_URL} ...")
    try:
        requests.get(BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach {BASE_URL}. Start the JS API first (npm start).")
        sys.exit(1)

    mirror_ok = test_mirror_invariance()
    test_direction_preserved()
    sys.exit(0 if mirror_ok else 1)
