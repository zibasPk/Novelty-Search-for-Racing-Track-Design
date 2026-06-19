"""
Visual comparison of simplex-noise vs uniform point generation.

For each seed it asks the JS track-generation API (mapElitesAPI) to generate a
Voronoi layout twice — once with uniform random sites and once with
simplex-noise-driven sites — and plots the sampled points together with the
Voronoi cells. The track spline itself is intentionally NOT drawn, so the focus
stays on how the two RNG modes spread their points and shape the cells.

Requires the JS API server running on localhost:4242:
    node sim/mapElitesAPI.js   (from the `src` folder)

Run:
    python qd/visualizations/simplex_vs_uniform_points.py

Produces in this directory:
    simplex_vs_uniform_points.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import requests

BASE_URL = "http://localhost:4242"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plots", "simplex_vs_uniform")
BBOX = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}

# rngMode values mirror src/utils/constants.js -> RngMode
RNG_UNIFORM = 0
RNG_SIMPLEX = 1  # internally named PERLIN, implemented with the simplex-noise lib

MODE = "voronoi"

# Number of seeds to compare (one random seed per row, re-rolled each run).
N_SEEDS = 4
TRACK_SIZE = 8

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_POINTS = "#2166ac"
COLOR_EDGES = "#bbbbbb"


# ─── API ─────────────────────────────────────────────────────────────────────

def generate(seed: int, rng_mode: int, track_size: int = TRACK_SIZE) -> dict:
    """Return the full generator JSON (dataSet, diagram, selectedCells)."""
    r = requests.post(
        f"{BASE_URL}/genforweb",
        json={
            "id": str(seed),
            "mode": MODE,
            "trackSize": track_size,
            "rngMode": rng_mode,
            # genforweb requires the field; null falls back to defaults for uniform.
            "perlin_parameters": None,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["generator"]


# ─── plotting ─────────────────────────────────────────────────────────────────

def _draw_generation(ax, gen, point_color, title):
    # Voronoi cells (all edges of the diagram).
    ex, ey = [], []
    for e in gen["diagram"]["edges"]:
        ex += [e["va"]["x"], e["vb"]["x"], np.nan]
        ey += [e["va"]["y"], e["vb"]["y"], np.nan]
    if ex:
        ax.plot(ex, ey, color=COLOR_EDGES, linewidth=0.6, zorder=1)

    # Generated points.
    ds = gen["dataSet"]
    if ds:
        ax.scatter([p["x"] for p in ds], [p["y"] for p in ds],
                   s=10, c=point_color, alpha=0.85, linewidths=0, zorder=3)

    ax.set_xlim(BBOX["xl"], BBOX["xr"])
    ax.set_ylim(BBOX["yb"], BBOX["yt"])  # flip y so it matches the web visualizer
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=4)


def _try_pair(seed):
    """Generate both modes for a seed; return None if the server can't build either."""
    try:
        return generate(seed, RNG_SIMPLEX), generate(seed, RNG_UNIFORM)
    except requests.exceptions.HTTPError:
        return None


def main():
    # Re-roll seeds each run, skipping any the server fails to generate.
    rows, seeds = [], []
    while len(rows) < N_SEEDS:
        seed = int(np.random.randint(0, 100000))
        pair = _try_pair(seed)
        if pair is not None:
            rows.append(pair)
            seeds.append(seed)
    print(f"Seeds: {seeds}")

    fig, axes = plt.subplots(N_SEEDS, 2, figsize=(7, 3.4 * N_SEEDS), squeeze=False)
    fig.suptitle(
        "Voronoi site generation — Simplex noise vs Uniform\n"
        "points and Voronoi cells, no track drawn",
        fontsize=11, y=1.005,
    )

    for row, (seed, (gen_simplex, gen_uniform)) in enumerate(zip(seeds, rows)):
        _draw_generation(axes[row][0], gen_simplex, COLOR_POINTS,
                         f"Seed {seed} — Simplex noise  (n={len(gen_simplex['dataSet'])})")
        _draw_generation(axes[row][1], gen_uniform, COLOR_POINTS,
                         f"Seed {seed} — Uniform  (n={len(gen_uniform['dataSet'])})")

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "simplex_vs_uniform_points.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to {BASE_URL} ...")
    try:
        requests.get(BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach {BASE_URL}. Start the JS API first "
              f"(node sim/mapElitesAPI.js from the src folder).")
        sys.exit(1)

    main()
