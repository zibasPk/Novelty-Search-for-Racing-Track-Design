"""
Generate and plot tracks for seeds 0–200 by calling the /genforweb API.
Displays dataset points, voronoi edges, and the smoothed track spline.
"""

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import math
import sys

API_URL = "http://localhost:4242/genforweb"
SEED_START = 0
SEED_END = 200
MODE = "voronoi"
TRACK_SIZE = 5

num_tracks = SEED_END - SEED_START + 1
cols = int(math.ceil(math.sqrt(num_tracks)))
rows = int(math.ceil(num_tracks / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=200)
axes = np.array(axes).flatten()

for idx, seed in enumerate(range(SEED_START, SEED_END + 1)):
    ax = axes[idx]
    print(f"Generating seed {seed} ({idx + 1}/{num_tracks})...", end=" ", flush=True)

    try:
        resp = requests.post(API_URL, json={
            "id": str(seed),
            "mode": MODE,
            "trackSize": TRACK_SIZE,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"FAILED: {e}")
        ax.set_title(f"#{seed} ERR", fontsize=6, color="red")
        ax.axis("off")
        continue

    gen = data["generator"]
    track = data["track"]

    # --- Voronoi edges (light grey) ---
    edges = gen["diagram"]["edges"]
    edge_lines = []
    for e in edges:
        va, vb = e["va"], e["vb"]
        edge_lines.append([(va["x"], va["y"]), (vb["x"], vb["y"])])
    lc = mc.LineCollection(edge_lines, colors="#cccccc", linewidths=0.4, zorder=1)
    ax.add_collection(lc)

    # --- Dataset points (blue dots) ---
    ds = gen["dataSet"]
    ds_x = [p["x"] for p in ds]
    ds_y = [p["y"] for p in ds]
    ax.scatter(ds_x, ds_y, s=1, c="steelblue", zorder=2, linewidths=0)

    # --- Highlight selected cells (fill) ---
    for cell in gen["selectedCells"]:
        # Build ordered polygon by following halfedge direction.
        # In rhill-voronoi-core, if the halfedge's site == edge.lSite
        # then start=va, end=vb; otherwise start=vb, end=va.
        site = cell["site"]
        ordered_pts = []
        for he in cell["halfedges"]:
            edge = he["edge"]
            lSite = edge.get("lSite")
            if lSite and lSite["x"] == site["x"] and lSite["y"] == site["y"]:
                ordered_pts.append((edge["va"]["x"], edge["va"]["y"]))
            else:
                ordered_pts.append((edge["vb"]["x"], edge["vb"]["y"]))
        if ordered_pts:
            poly = MplPolygon(ordered_pts, closed=True, alpha=0.15, fc="orange", ec="none", zorder=1)
            ax.add_patch(poly)

    # --- Track spline (red line) ---
    tx = [p["x"] for p in track]
    ty = [p["y"] for p in track]
    # Close the loop
    tx.append(tx[0])
    ty.append(ty[0])
    ax.plot(tx, ty, color="crimson", linewidth=1.0, zorder=3)

    # --- Formatting ---
    bbox = gen["bbox"]
    ax.set_xlim(bbox["xl"], bbox["xr"])
    ax.set_ylim(bbox["yb"], bbox["yt"])  # y-axis: top=0 in screen coords
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"#{seed}", fontsize=5, pad=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    print("OK")

# Hide unused subplots
for idx in range(num_tracks, len(axes)):
    axes[idx].axis("off")

plt.suptitle(f"Tracks (seeds {SEED_START}–{SEED_END}), mode={MODE}, trackSize={TRACK_SIZE}",
             fontsize=10, y=1.0)
plt.tight_layout()
plt.savefig("tracks_overview.png", dpi=250, bbox_inches="tight")
print(f"\nSaved tracks_overview.png ({rows}x{cols} grid)")
