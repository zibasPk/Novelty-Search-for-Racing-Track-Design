"""
Visual illustration of the two genetic operators used by the MAP-Elites emitter:
mutation and crossover.

For a Voronoi individual the JS track-generation API (mapElitesAPI) exposes the
same operators the emitter uses:
    /genforweb   build a parent (points, Voronoi cells, selected sites, track)
    /mutate      perturb a parent into a mutant
    /crossover   recombine two parents into an offspring
    /reconstruct rebuild the track/diagram for a mutant or offspring so it can
                 be drawn

Each panel shows the generated points, the Voronoi cells, the selected sites
(the cells whose sites define the track) highlighted, and the resulting track.

Requires the JS API server running on localhost:4242:
    node sim/mapElitesAPI.js   (from the `src` folder)

Run (uses the default seeds/sizes, or override positionally):
    python qd/visualizations/mutation_crossover.py
    python qd/visualizations/mutation_crossover.py 42 123 777
    # mut_seed  cx_seed_a cx_seed_b  cx_size_a cx_size_b
    python qd/visualizations/mutation_crossover.py 42 123 777 5 9

Produces in plots/mutation_crossover:
    mutation.png    parent  -> mutant
    crossover.png   parent A + parent B -> offspring
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import requests

BASE_URL = "http://localhost:4242"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plots", "mutation_crossover")
BBOX = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}

MODE = "voronoi"
RNG_UNIFORM = 0
TRACK_SIZE = 8
INTENSITY_MUTATION = 30

# Default seeds: one parent for mutation, two parents for crossover.
MUT_SEED = 85
# CX_SEEDS = (67, 18)
CX_SEEDS = (297, 707)
# Track size (number of selected sites) for each crossover parent.
CX_SIZES = (8, 7)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_POINTS = "#9ecae1"   # all candidate sites (background)
COLOR_EDGES = "#bbbbbb"    # Voronoi cell edges
COLOR_SELECTED = "#2166ac"  # selected sites that define the track
COLOR_TRACK = "#d6604d"    # smoothed track spline


# ─── API ─────────────────────────────────────────────────────────────────────

def _sites(selected_cells):
    """Reduce raw Voronoi cells (or already-flat sites) to plain {x, y} sites."""
    return [{"x": c["site"]["x"], "y": c["site"]["y"]} if "site" in c
            else {"x": c["x"], "y": c["y"]} for c in selected_cells]


def generate(seed: int, track_size: int = TRACK_SIZE) -> dict:
    """Build a parent individual and return both its solution dict and panel data."""
    r = requests.post(
        f"{BASE_URL}/genforweb",
        json={"id": str(seed), "mode": MODE, "trackSize": track_size,
              "rngMode": RNG_UNIFORM, "perlin_parameters": None,
              # Keep the track in the same frame as the points/cells for drawing.
              "canonicalize": False},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()
    gen = res["generator"]
    sites = _sites(gen["selectedCells"])
    individual = {
        "id": seed, "mode": MODE, "trackSize": gen["trackSize"],
        "dataSet": gen["dataSet"], "selectedCells": sites, "rngMode": RNG_UNIFORM,
    }
    panel = {
        "edges": gen["diagram"]["edges"],
        "points": gen["dataSet"],
        "sites": sites,
        "track": res.get("track") or [],
    }
    return individual, panel


def reconstruct(seed, data_set, selected_cells) -> dict:
    """Rebuild a child's track/diagram from its points + selected sites."""
    r = requests.post(
        f"{BASE_URL}/reconstruct",
        json={"mode": MODE, "seed": seed, "dataSet": data_set,
              "selectedCells": selected_cells, "trackSize": len(selected_cells),
              "canonicalize": False},
        timeout=60,
    )
    r.raise_for_status()
    rc = r.json()
    return {
        "edges": rc["edges"],
        "points": rc["dataSet"],
        "sites": _sites(rc["selectedCells"]),
        "track": rc.get("track") or [],
    }


def mutate(individual, genetic_seed) -> dict:
    r = requests.post(
        f"{BASE_URL}/mutate",
        json={"individual": individual, "intensityMutation": INTENSITY_MUTATION,
              "genetic_seed": genetic_seed},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["mutated"]


def crossover(parent1, parent2, genetic_seed) -> dict:
    r = requests.post(
        f"{BASE_URL}/crossover",
        json={"mode": MODE, "parent1": parent1, "parent2": parent2,
              "genetic_seed": genetic_seed},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["offspring"]


# ─── plotting ─────────────────────────────────────────────────────────────────

def _draw_panel(ax, panel, title):
    # Voronoi cells.
    ex, ey = [], []
    for e in panel["edges"]:
        ex += [e["va"]["x"], e["vb"]["x"], np.nan]
        ey += [e["va"]["y"], e["vb"]["y"], np.nan]
    if ex:
        ax.plot(ex, ey, color=COLOR_EDGES, linewidth=0.6, zorder=1)

    # Track spline (closed loop), drawn unclipped so any overhang stays visible.
    track = panel["track"]
    if track:
        ax.add_patch(plt.Rectangle(
            (BBOX["xl"], BBOX["yt"]),
            BBOX["xr"] - BBOX["xl"], BBOX["yb"] - BBOX["yt"],
            fill=False, edgecolor=COLOR_EDGES, linewidth=0.8, zorder=1.5))
        tx = [p["x"] for p in track] + [track[0]["x"]]
        ty = [p["y"] for p in track] + [track[0]["y"]]
        ax.plot(tx, ty, color=COLOR_TRACK, linewidth=1.6,
                solid_capstyle="round", zorder=2, clip_on=False)

    # All candidate points (background) then the selected sites (highlighted).
    pts = panel["points"]
    if pts:
        ax.scatter([p["x"] for p in pts], [p["y"] for p in pts],
                   s=8, c=COLOR_POINTS, alpha=0.7, linewidths=0, zorder=3)
    sites = panel["sites"]
    if sites:
        ax.scatter([p["x"] for p in sites], [p["y"] for p in sites],
                   s=20, c=COLOR_SELECTED, edgecolors="white", linewidths=0.5,
                   zorder=4)

    # View limits: the domain, widened to include any track overhang.
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
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=4)


def _save(fig, filename):
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def render_mutation(seed):
    print(f"Mutation - parent seed {seed}")
    individual, parent_panel = generate(seed)
    mutated = mutate(individual, genetic_seed=0.5)
    mutant_panel = reconstruct(seed, mutated["dataSet"], _sites(mutated["selectedCells"]))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), squeeze=False)
    _draw_panel(axes[0][0], parent_panel, "Parent")
    _draw_panel(axes[0][1], mutant_panel, "Mutant")
    _save(fig, "mutation.png")


def render_crossover(seed_a, seed_b, size_a=TRACK_SIZE, size_b=TRACK_SIZE):
    print(f"Crossover - parent seeds {seed_a}, {seed_b} "
          f"(sizes {size_a}, {size_b})")
    ind_a, panel_a = generate(seed_a, track_size=size_a)
    ind_b, panel_b = generate(seed_b, track_size=size_b)
    offspring = crossover(ind_a, ind_b, genetic_seed=0.7)
    child_panel = reconstruct(seed_a, offspring["ds"], _sites(offspring["sel"]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), squeeze=False)
    _draw_panel(axes[0][0], panel_a, "Parent A")
    _draw_panel(axes[0][1], panel_b, "Parent B")
    _draw_panel(axes[0][2], child_panel, "Offspring")
    _save(fig, "crossover.png")


def main(args):
    # Positional args: mut_seed  cx_seed_a cx_seed_b  [cx_size_a cx_size_b]
    mut_seed = args[0] if len(args) >= 1 else MUT_SEED
    cx_seeds = (args[1], args[2]) if len(args) >= 3 else CX_SEEDS
    cx_sizes = (args[3], args[4]) if len(args) >= 5 else CX_SIZES
    render_mutation(mut_seed)
    render_crossover(*cx_seeds, *cx_sizes)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to {BASE_URL} ...")
    try:
        requests.get(BASE_URL, timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot reach {BASE_URL}. Start the JS API first "
              f"(node sim/mapElitesAPI.js from the src folder).")
        sys.exit(1)

    main([int(a) for a in sys.argv[1:]])
