"""
Visual comparison of simplex-noise vs uniform point generation.

For a single seed it asks the JS track-generation API (mapElitesAPI) to generate
a Voronoi layout twice — once with uniform random sites and once with
simplex-noise-driven sites — and plots the result at three increasing levels of
detail:
    1. points only
    2. points + Voronoi cells
    3. points + Voronoi cells + the overlaid track

Requires the JS API server running on localhost:4242:
    node sim/mapElitesAPI.js   (from the `src` folder)

Run (uses the SEEDS list, or pass seeds as args):
    python qd/analysis/concept_figure_gen/simplex_vs_uniform_points.py
    python qd/analysis/concept_figure_gen/simplex_vs_uniform_points.py 42 123 999

Produces, per seed, in data/plots/simplex_vs_uniform (seed in the filename):
    simplex_vs_uniform_points_seed<n>.png    (level 1 — points only)
    simplex_vs_uniform_voronoi_seed<n>.png   (level 2 — points + Voronoi cells)
    simplex_vs_uniform_track_seed<n>.png      (level 3 — + overlaid track)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import requests

BASE_URL = "http://localhost:4242"
OUT_DIR = os.path.join("data", "plots", "simplex_vs_uniform")
BBOX = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}

# rngMode values mirror src/utils/constants.js -> RngMode
RNG_UNIFORM = 0
RNG_SIMPLEX = 1  # internally named PERLIN, implemented with the simplex-noise lib

MODE = "voronoi"
# Seeds to compare — one set of plots is produced per seed so the best can be
# picked by eye. Edit this list (or pass seeds on the command line).
SEEDS = [10069]
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
COLOR_TRACK = "#d6604d"


# ─── API ─────────────────────────────────────────────────────────────────────

def generate(seed: int, rng_mode: int, track_size: int = TRACK_SIZE) -> dict:
    """Return the genforweb JSON (generator metadata + smoothed track spline)."""
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
    return r.json()


# ─── plotting ─────────────────────────────────────────────────────────────────

def _draw_generation(ax, result, point_color, show_edges, show_track):
    gen = result["generator"]

    # Voronoi cells (all edges of the diagram).
    if show_edges:
        ex, ey = [], []
        for e in gen["diagram"]["edges"]:
            ex += [e["va"]["x"], e["vb"]["x"], np.nan]
            ey += [e["va"]["y"], e["vb"]["y"], np.nan]
        if ex:
            ax.plot(ex, ey, color=COLOR_EDGES, linewidth=0.6, zorder=1)

    # Overlaid track spline (closed loop). Drawn unclipped so geometry that spills
    # past the generation domain stays visible, and the domain edge is marked
    # explicitly so the overlap is obvious.
    track = result.get("track") or [] if show_track else []
    if track:
        ax.add_patch(plt.Rectangle(
            (BBOX["xl"], BBOX["yt"]),
            BBOX["xr"] - BBOX["xl"], BBOX["yb"] - BBOX["yt"],
            fill=False, edgecolor=COLOR_EDGES, linewidth=0.8, zorder=1.5))
        tx = [p["x"] for p in track] + [track[0]["x"]]
        ty = [p["y"] for p in track] + [track[0]["y"]]
        ax.plot(tx, ty, color=COLOR_TRACK, linewidth=1.6,
                solid_capstyle="round", zorder=2, clip_on=False)

    # Generated points.
    ds = gen["dataSet"]
    if ds:
        ax.scatter([p["x"] for p in ds], [p["y"] for p in ds],
                   s=10, c=point_color, alpha=0.85, linewidths=0, zorder=3)

    # View limits: the generation domain, widened to include any track overhang.
    x_lo, x_hi = BBOX["xl"], BBOX["xr"]
    y_lo, y_hi = BBOX["yt"], BBOX["yb"]
    if track:
        pad = 20
        xs = [p["x"] for p in track]
        ys = [p["y"] for p in track]
        x_lo, x_hi = min(x_lo, min(xs) - pad), max(x_hi, max(xs) + pad)
        y_lo, y_hi = min(y_lo, min(ys) - pad), max(y_hi, max(ys) + pad)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)  # flip y so it matches the web visualizer
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    # Without the Voronoi cells there is nothing delimiting the domain, so frame
    # the plot with a boundary box instead.
    show_boundary = not show_edges
    for spine in ax.spines.values():
        spine.set_visible(show_boundary)
        if show_boundary:
            spine.set_edgecolor(COLOR_EDGES)
            spine.set_linewidth(0.8)


# Three levels of detail (filename, show_edges, show_track), each its own figure.
LEVELS = [
    ("simplex_vs_uniform_points.png",  False, False),
    ("simplex_vs_uniform_voronoi.png", True,  False),
    ("simplex_vs_uniform_track.png",   True,  True),
]


def _render_level(res_simplex, res_uniform, filename, show_edges, show_track):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), squeeze=False)

    _draw_generation(axes[0][0], res_simplex, COLOR_POINTS,
                     show_edges, show_track)
    _draw_generation(axes[0][1], res_uniform, COLOR_POINTS,
                     show_edges, show_track)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def render_seed(seed: int):
    """Produce the three-level comparison for a single seed (seed in the filename)."""
    print(f"Seed: {seed}")
    res_simplex = generate(seed, RNG_SIMPLEX)
    res_uniform = generate(seed, RNG_UNIFORM)

    for filename, show_edges, show_track in LEVELS:
        seeded = filename.replace(".png", f"_seed{seed}.png")
        _render_level(res_simplex, res_uniform, seeded, show_edges, show_track)


def main(seeds):
    for seed in seeds:
        try:
            render_seed(seed)
        except requests.exceptions.HTTPError as e:
            print(f"  skipped seed {seed}: {e}")


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to {BASE_URL} ...")
    try:
        requests.get(BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach {BASE_URL}. Start the JS API first "
              f"(node sim/mapElitesAPI.js from the src folder).")
        sys.exit(1)

    # Optional: seeds passed as command-line args override the SEEDS list.
    cli_seeds = [int(a) for a in sys.argv[1:]]
    main(cli_seeds or SEEDS)
