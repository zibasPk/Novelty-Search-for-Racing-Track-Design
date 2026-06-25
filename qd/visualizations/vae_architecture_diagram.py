"""Render schematic diagrams of the MetricsVAE Circular-CNN architecture.

Run directly with ``python qd/visualizations/vae_architecture_diagram.py``
(or ``python vae_architecture_diagram.py`` from this folder). Produces
several PNGs in ``qd/visualizations/plots/``:

- ``qd_pipeline.png``         — whole unsupervised QD loop (AURORA-style)
- ``vae_architecture.png``   — full encoder/decoder pipeline overview
- ``circular_resblock.png``   — internals of a single CircularResBlock
- ``circular_padding.png``    — circular ("wrap-around") padding concept
"""

import math
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath

# Allow running this file directly (not as ``python -m ...``) by putting the
# project root — three levels up from qd/visualizations/ — on the import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qd.vae.config import MODEL_CONFIG

matplotlib.use("Agg")

# ── Style ──────────────────────────────────────────────────────────────────

DX, DY = 0.45, 0.32          # isometric "depth" offset for the pseudo-3D faces
ARROW_COLOR = "#2f6f6f"
EDGE_COLOR = "#3a3a3a"

ENCODER_FACE = "#f7cf8b"
ENCODER_ACCENT = "#e0792e"
LATENT_FACE = "#9fc3e8"
DECODER_FACE = "#bfe0e6"
OUTPUT_FACE = "#caa6d8"
NORM_FACE = "#e3e3e3"

OUT_DIR = os.path.join("qd", "visualizations", "plots", "vae_architecture")


def shade(color, factor):
    """factor > 1 lightens (towards white), factor < 1 darkens (towards black)."""
    r, g, b = mcolors.to_rgb(color)
    if factor >= 1:
        t = factor - 1
        r, g, b = r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t
    else:
        r, g, b = r * factor, g * factor, b * factor
    return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1))


def draw_cuboid(ax, x, y, w, h, facecolor, dx=DX, dy=DY, lw=1.3, zorder=3):
    """Draw a pseudo-3D box (front + top + side faces). Returns (right_x, top_y)."""
    front = Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=EDGE_COLOR, lw=lw, zorder=zorder)
    top = Polygon(
        [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
        facecolor=shade(facecolor, 1.35), edgecolor=EDGE_COLOR, lw=lw, zorder=zorder,
    )
    side = Polygon(
        [(x + w, y), (x + w, y + h), (x + w + dx, y + h + dy), (x + w + dx, y + dy)],
        facecolor=shade(facecolor, 0.7), edgecolor=EDGE_COLOR, lw=lw, zorder=zorder,
    )
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)
    return x + w + dx, y + h + dy


def draw_stack(ax, x, y, w, h, facecolor, n=4, step=0.22):
    """Draw n overlapping cuboids (back-to-front) representing repeated layers."""
    right, top = x + w + DX, y + h + DY
    for i in reversed(range(n)):
        ox, oy = i * step, i * step * (DY / DX)
        r, t = draw_cuboid(ax, x + ox, y + oy, w, h, facecolor)
        right, top = max(right, r), max(top, t)
    return right, top


def label(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 10)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, text, **kw)


def arrow(ax, x0, x1, y=0.0, **kw):
    style = dict(arrowstyle="-|>", mutation_scale=18, color=ARROW_COLOR, lw=2, zorder=4)
    style.update(kw)
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), **style))


def dashed(ax, p0, p1):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color="#888888", lw=1, zorder=2)


def draw_box(ax, x, y, w, h, text, facecolor, fontsize=10, zorder=3):
    """Flat rounded box used for flowchart-style diagrams."""
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=facecolor, edgecolor=EDGE_COLOR, lw=1.3, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=zorder + 1)
    return x + w, y + h


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved diagram to {out_path}")


# ── Whole QD pipeline (shared node icons) ─────────────────────────────────────
#
# Several candidate layouts share the same five "nodes" (Archive, VAE encoder,
# Track, Track metrics, Dataset) drawn as small icons centred on a point. Each
# helper draws one icon at ``c`` and returns a rough (half_w, half_h) footprint
# so the layout functions can aim arrows at the icon edges.

GREY = "#9a9a9a"
SUBTLE = "#555555"
VAE_LAYER_SPECS = [(6, ENCODER_FACE, 0.0), (4, "#c9c98a", 0.95), (2, "#9cc79a", 1.9)]


def qd_arrow(ax, p0, p1, rad=0.0, color=GREY, lw=3.0, arrowstyle="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle, mutation_scale=24, color=color, lw=lw, zorder=2,
    ))


def cubic(p0, c1, c2, p3):
    """A cubic-Bézier path through p0 → p3 with the two given control points."""
    return MplPath([p0, c1, c2, p3],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])


def qd_label(ax, x, y, text, color="black", **kw):
    kw.setdefault("fontsize", 11)
    kw.setdefault("fontweight", "bold")
    label(ax, x, y, text, color=color, **kw)


# Blue → purple → red ramp for the repertoire cloud (no pale/white centre,
# unlike a diverging map such as Spectral or coolwarm).
ARCHIVE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "archive_blue_red", ["#2166ac", "#8e44ad", "#b2182b"],
)


def draw_archive(ax, c, rng, rx=2.2, ry=1.15):
    cx, cy = c
    n = 1300
    r = np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    px = cx + rx * r * np.cos(th)
    py = cy + ry * r * np.sin(th)
    ax.scatter(px, py, c=px - py, cmap=ARCHIVE_CMAP, s=6, alpha=0.85, zorder=3)
    label(ax, cx, cy + ry + 0.45, "Archive", fontsize=15, fontweight="bold")
    label(ax, cx, cy + ry + 0.12, "repertoire of elite tracks", fontsize=9, color=SUBTLE)
    return rx, ry


def draw_vae(ax, c):
    """VAE encoder funnel. Footprint spans vy..vy+1.9 vertically, ~1.4 wide."""
    vx, vy = c
    coords, colors = [], []
    for count, color, dy in VAE_LAYER_SPECS:
        coords.append([(vx + (j - (count - 1) / 2) * 0.5, vy + dy) for j in range(count)])
        colors.append(color)
    for lower, upper in zip(coords, coords[1:]):
        for x0, y0 in lower:
            for x1, y1 in upper:
                ax.plot([x0, x1], [y0, y1], color=ENCODER_ACCENT, lw=0.5, alpha=0.4, zorder=3)
    for layer, color in zip(coords, colors):
        for x, y in layer:
            ax.add_patch(Circle((x, y), 0.2, facecolor=color, edgecolor=EDGE_COLOR, lw=1.0, zorder=4))
    label(ax, vx, vy - 0.55, "Pretrained VAE encoder", fontsize=13, fontweight="bold")
    label(ax, vx, vy - 0.9, "dimensionality reduction", fontsize=8.5, color=SUBTLE)
    return 1.5, 1.0


def draw_track(ax, c):
    tx, ty = c
    t = np.linspace(0, 2 * np.pi, 240)
    rad = 0.85 + 0.18 * np.sin(3 * t) + 0.1 * np.cos(5 * t)
    X, Y = tx + rad * np.cos(t), ty + rad * np.sin(t) * 0.85
    ax.plot(X, Y, color=EDGE_COLOR, lw=4, zorder=3)
    ax.plot(X, Y, color=ENCODER_FACE, lw=1.6, zorder=3)
    label(ax, tx, ty + 1.4, "Track", fontsize=13, fontweight="bold")
    label(ax, tx, ty - 1.45, "candidate track\n(Voronoi / spline)", fontsize=8.5, color=SUBTLE)
    return 1.05, 0.95


def draw_metrics(ax, c, rng):
    mx, my = c
    sx = np.linspace(-1.2, 1.2, 140)
    sy = 0.3 * np.sin(4 * sx) + 0.15 * np.sin(9 * sx + 1) + 0.07 * rng.standard_normal(140)
    sy = np.convolve(sy, np.ones(5) / 5, mode="same")
    ax.plot(mx + sx, my + sy, color="#7fa86b", lw=2.5, zorder=3)
    ax.add_patch(FancyArrowPatch((mx - 1.35, my - 0.55), (mx + 1.45, my - 0.55),
                                 arrowstyle="-|>", mutation_scale=12, color=EDGE_COLOR, lw=1.2, zorder=3))
    ax.add_patch(FancyArrowPatch((mx - 1.35, my - 0.55), (mx - 1.35, my + 0.6),
                                 arrowstyle="-|>", mutation_scale=12, color=EDGE_COLOR, lw=1.2, zorder=3))
    label(ax, mx, my - 0.95, "Track metrics", fontsize=13, fontweight="bold")
    label(ax, mx, my - 1.28, "speed · steering · position", fontsize=8.5, color=SUBTLE)
    return 1.45, 0.9


def draw_dataset(ax, c):
    dx, dy = c
    cmap = plt.get_cmap("Spectral")
    n_lines = 6
    lx = np.linspace(-1.0, 1.0, 120)
    for i in range(n_lines):
        ly = 0.18 * np.sin(5 * lx + i) + 0.08 * np.cos(8 * lx + 2 * i)
        ax.plot(dx + lx, dy + (i - n_lines / 2) * 0.3 + ly * 0.5, color=cmap(i / n_lines), lw=1.8, zorder=3)
    label(ax, dx, dy + 1.25, "Dataset", fontsize=13, fontweight="bold")
    label(ax, dx, dy - 1.2, "elite metric sequences", fontsize=8.5, color=SUBTLE)
    return 1.1, 1.0


# ── Whole QD pipeline (horizontal assembly line) ──────────────────────────────

def build_qd_pipeline(cfg):
    """Schematic of the whole unsupervised QD loop as a horizontal assembly line.

    All five nodes sit on one baseline and read left-to-right
    (Archive → Track → Track metrics → Pretrained VAE encoder); the loop is
    closed by a ``descriptor`` arc over the top and a periodic fine-tuning
    branch (Archive → Dataset → VAE) running underneath.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(21, 9.5))

    archive_c = (2.6, 6.0)
    track_c = (8.0, 6.0)
    metric_c = (12.8, 6.0)
    vae_c = (17.6, 5.1)
    data_c = (9.8, 1.0)

    a_rx, a_ry = draw_archive(ax, archive_c, rng)

    # self-loop on the left of the archive: the repertoire is periodically
    # re-projected onto itself with the freshly fine-tuned descriptor space.
    loop_top = (archive_c[0] - a_rx - 0.1, archive_c[1] + 0.4)
    loop_bot = (archive_c[0] - a_rx - 0.1, archive_c[1] - 0.4)
    ax.add_patch(FancyArrowPatch(
        path=cubic(loop_top, (archive_c[0] - a_rx - 1.7, archive_c[1] + 1.1),
                   (archive_c[0] - a_rx - 1.7, archive_c[1] - 1.1), loop_bot),
        arrowstyle="-|>", mutation_scale=24, color=GREY, lw=3.0, zorder=2))
    qd_label(ax, archive_c[0] - a_rx - 0.75, archive_c[1] - 1.5,
             "periodical\nremapping", fontsize=10)

    draw_track(ax, track_c)
    draw_metrics(ax, metric_c, rng)
    draw_vae(ax, vae_c)
    draw_dataset(ax, data_c)

    # main left-to-right chain
    qd_arrow(ax, (archive_c[0] + a_rx + 0.2, 6.0), (track_c[0] - 1.35, 6.0))
    qd_label(ax, 5.9, 6.5, "select, mutate\n& crossover", fontsize=10)
    qd_arrow(ax, (track_c[0] + 1.35, 6.0), (metric_c[0] - 1.75, 6.0))
    qd_label(ax, 10.4, 6.4, "simulate", fontsize=10)
    qd_arrow(ax, (metric_c[0] + 1.75, 6.0), (vae_c[0] - 1.0, 6.0), rad=0.0)
    qd_label(ax, 15.4, 6.45, "encode", fontsize=10)

    # Behavioural Descriptor (from VAE) and fitness (from metrics) branch up and
    # merge tangentially into one trunk that flows back into the archive.
    trunk_y = 9.0
    j_metrics = (9.5, trunk_y)
    j_vae = (12.5, trunk_y)
    vae_src = (vae_c[0], vae_c[1] + 2.3)
    met_src = (metric_c[0] - 0.2, 6.95)
    archive_end = (archive_c[0], archive_c[1] + a_ry + 0.8)

    # feeders: leave the source vertically, arrive at the trunk horizontally
    ax.add_patch(PathPatch(
        cubic(met_src, (met_src[0], met_src[1] + 1.7), (j_metrics[0] + 1.4, trunk_y), j_metrics),
        fill=False, edgecolor=GREY, lw=3.0, zorder=2, capstyle="round"))
    ax.add_patch(PathPatch(
        cubic(vae_src, (vae_src[0], trunk_y), (j_vae[0] + 2.3, trunk_y), j_vae),
        fill=False, edgecolor=GREY, lw=3.0, zorder=2, capstyle="round"))
    # shared trunk between the two joins, then one arrow swooping into the archive
    ax.plot([j_vae[0], j_metrics[0]], [trunk_y, trunk_y], color=GREY, lw=3.0, zorder=2,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        path=cubic(j_metrics, (6.0, trunk_y), (3.4, 8.4), archive_end),
        arrowstyle="-|>", mutation_scale=24, color=GREY, lw=3.0, zorder=2))
    qd_label(ax, 12.2, 9.4, "Behavioural Descriptor", fontsize=11)
    qd_label(ax, 11.2, 7.9, "fitness", fontsize=10)

    # periodic fine-tuning branch under the bottom
    qd_arrow(ax, (archive_c[0], archive_c[1] - a_ry - 0.2), (data_c[0] - 1.4, data_c[1] + 0.4), rad=0.22)
    qd_label(ax, 5.0, 3.1, "elites", fontsize=10)
    qd_arrow(ax, (data_c[0] + 1.4, data_c[1]), (vae_c[0], vae_c[1] - 1.25), rad=-0.16)
    qd_label(ax, 14.6, 1.9, "fine-tune", fontsize=10)

    ax.set_xlim(-1.8, 20.4)
    ax.set_ylim(-0.8, 10.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ── Whole QD pipeline (vertical, half-page portrait) ──────────────────────────

def build_qd_pipeline_vertical(cfg):
    """Portrait version of :func:`build_qd_pipeline` for a half-page figure.

    The five nodes are stacked top-to-bottom in a single column
    (Archive → Track → Track metrics → Pretrained VAE encoder), reading
    downward. The loop is closed by a Behavioural-Descriptor + fitness trunk
    running up the right-hand side back into the archive, with the periodic
    fine-tuning branch (Archive → Dataset → VAE) running down the left.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 13.5))

    # white halo behind edge labels so the arrow line is "broken" under the text
    gap = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none")

    cx = 4.0                         # shared centre column for the main chain
    archive_c = (cx, 19.2)
    track_c = (cx, 14.7)
    metric_c = (cx, 10.2)
    vae_c = (cx, 4.7)                # encoder funnel grows upward to vy + 1.9
    data_c = (-0.2, 12.2)

    a_rx, a_ry = draw_archive(ax, archive_c, rng)

    # self-loop on the left of the archive: the repertoire is periodically
    # re-projected onto itself with the freshly fine-tuned descriptor space.
    loop_top = (archive_c[0] - a_rx - 0.1, archive_c[1] + 0.4)
    loop_bot = (archive_c[0] - a_rx - 0.1, archive_c[1] - 0.4)
    ax.add_patch(FancyArrowPatch(
        path=cubic(loop_top, (archive_c[0] - a_rx - 1.7, archive_c[1] + 1.1),
                   (archive_c[0] - a_rx - 1.7, archive_c[1] - 1.1), loop_bot),
        arrowstyle="-|>", mutation_scale=24, color=GREY, lw=3.0, zorder=2))
    qd_label(ax, archive_c[0] - a_rx - 0.8, archive_c[1],
             "periodical\nremapping", fontsize=10, bbox=gap)

    draw_track(ax, track_c)
    draw_metrics(ax, metric_c, rng)
    draw_vae(ax, vae_c)
    draw_dataset(ax, data_c)

    # main top-to-bottom chain (labels sit to the right of each arrow)
    qd_arrow(ax, (cx, archive_c[1] - a_ry - 0.2), (cx, track_c[1] + 1.75))
    qd_label(ax, cx + 1.75, 16.9, "select, mutate\n& crossover", fontsize=10, bbox=gap)
    qd_arrow(ax, (cx, track_c[1] - 1.95), (cx, metric_c[1] + 0.95))
    qd_label(ax, cx, 11.95, "simulate", fontsize=10, bbox=gap)
    qd_arrow(ax, (cx, metric_c[1] - 1.75), (cx, vae_c[1] + 2.15))
    qd_label(ax, cx, 7.65, "encode", fontsize=10, bbox=gap)

    # Behavioural Descriptor (from VAE) and fitness (from metrics) feed a shared
    # trunk that runs up the right side and swoops back into the archive.
    trunk_x = 8.6
    met_src = (metric_c[0] + 1.55, metric_c[1] + 0.1)
    vae_src = (vae_c[0] + 0.45, vae_c[1] + 2.0)
    j_metrics = (trunk_x, metric_c[1] + 1.1)
    j_vae = (trunk_x, vae_c[1] + 2.5)

    # feeders: leave the source horizontally, arrive at the trunk vertically
    ax.add_patch(PathPatch(
        cubic(met_src, (met_src[0] + 1.6, met_src[1]), (trunk_x, j_metrics[1] - 0.9), j_metrics),
        fill=False, edgecolor=GREY, lw=3.0, zorder=2, capstyle="round"))
    ax.add_patch(PathPatch(
        cubic(vae_src, (vae_src[0] + 1.0, vae_src[1] + 0.4), (trunk_x, j_vae[1] - 1.0), j_vae),
        fill=False, edgecolor=GREY, lw=3.0, zorder=2, capstyle="round"))
    # shared trunk between the two joins, then one arrow swooping into the archive
    ax.plot([trunk_x, trunk_x], [j_vae[1], 18.9], color=GREY, lw=3.0, zorder=2,
            solid_capstyle="round")
    archive_end = (archive_c[0] + a_rx - 0.2, archive_c[1] + 0.35)
    ax.add_patch(FancyArrowPatch(
        path=cubic((trunk_x, 18.9), (trunk_x, 20.3), (archive_c[0] + a_rx + 1.1, 20.3), archive_end),
        arrowstyle="-|>", mutation_scale=24, color=GREY, lw=3.0, zorder=2))
    # "Behavioural Descriptor" sits on the descriptor-only trunk segment,
    # between where it leaves the VAE and where the fitness feeder joins.
    qd_label(ax, trunk_x, (j_vae[1] + j_metrics[1]) / 2, "Behavioural\nDescriptor", fontsize=11, bbox=gap)
    qd_label(ax, 7.65, 10.5, "fitness", fontsize=10, bbox=gap)

    # periodic fine-tuning branch down the left: Archive → Dataset → VAE
    qd_arrow(ax, (archive_c[0] - 1.0, archive_c[1] - a_ry - 0.1), (data_c[0], data_c[1] + 1.5), rad=0.28)
    qd_label(ax, 0.8, 16.25, "elites", fontsize=10, bbox=gap)
    qd_arrow(ax, (data_c[0], data_c[1] - 1.4), (vae_c[0] - 1.35, vae_c[1] + 0.35), rad=0.28)
    qd_label(ax, 0.45, 7.55, "fine-tune", fontsize=10, bbox=gap)

    ax.set_xlim(-2.4, 10.6)
    ax.set_ylim(2.6, 21.8)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ── Full pipeline overview ───────────────────────────────────────────────────

def build_full_pipeline(cfg):
    fig, ax = plt.subplots(figsize=(22, 7.5))

    cursor = 0.0
    gap = 1.0

    # Input [T, 3]
    w, h = 1.6, 0.7
    right, top = draw_cuboid(ax, cursor, -h / 2, w, h, ENCODER_FACE)
    draw_cuboid(ax, cursor + w - 0.18, -h / 2, 0.18, h, ENCODER_ACCENT)
    label(ax, cursor + w / 2, top + 0.35, "Input", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, "[T, 3]\nspeed, steer, pos")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # input_projection: Linear(3 -> 128) -> [T, 128]
    w, h = 1.8, 3.0
    right, top = draw_cuboid(ax, cursor, -h / 2, w, h, ENCODER_FACE)
    label(ax, cursor + w / 2, top + 0.35, "input_projection\nLinear(3→128)", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, f"[T, {cfg['hidden_dim']}]")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # Encoder CircularResBlocks x n_layers -> [T, 128]
    w, h = 1.1, 3.0
    n = cfg["n_layers"]
    right, top = draw_stack(ax, cursor, -h / 2, w, h, ENCODER_FACE, n=n)
    dils = ", ".join(str(2 ** i) for i in range(n))
    label(ax, cursor + 1.1, top + 0.35, f"CircularResBlock ×{n}\ndilation {dils}", fontweight="bold")
    label(ax, cursor + 1.1, -h / 2 - 0.35, f"[T, {cfg['hidden_dim']}]")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # DFT power pool -> [K, 128]
    w, h = 1.4, 3.0
    right, top = draw_cuboid(ax, cursor, -h / 2, w, h, ENCODER_FACE)
    draw_cuboid(ax, cursor + w - 0.18, -h / 2, 0.18, h, ENCODER_ACCENT)
    label(ax, cursor + w / 2, top + 0.35, f"DFT Power Pool\nK={cfg['freq_bins']}", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, f"[{cfg['freq_bins']}, {cfg['hidden_dim']}]")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # fc_mu / fc_var -> mu [32], log_var [32]
    x0 = cursor
    w, h = 1.0, 1.0
    right_mu, top_mu = draw_cuboid(ax, x0, 0.4, w, h, LATENT_FACE)
    right_var, top_var = draw_cuboid(ax, x0, -h - 0.4, w, h, LATENT_FACE)
    label(ax, x0 + w / 2, top_mu + 0.35, f"fc_mu\nμ [{cfg['latent_dim']}]", fontweight="bold")
    label(ax, x0 + w / 2, -h - 0.4 - 0.35, f"fc_var\nσ² [{cfg['latent_dim']}]", fontweight="bold")
    right = max(right_mu, right_var)
    cursor = right + gap

    # reparameterize -> z [32]
    w, h = 0.9, 1.2
    right_z, top_z = draw_cuboid(ax, cursor, -h / 2, w, h, LATENT_FACE)
    label(ax, cursor + w / 2, top_z + 0.35, "reparameterize\nz ~ N(μ, σ²)", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, f"[{cfg['latent_dim']}]")
    dashed(ax, (x0 + 1.0, 0.4 + 0.5), (cursor, 0.5))
    dashed(ax, (x0 + 1.0, -1.4 + 0.5), (cursor, -0.5))
    cursor = right_z + gap
    arrow(ax, cursor - gap, cursor)

    # decoder fc + broadcast + positional encoding -> [T, 128]
    w, h = 1.8, 3.0
    right, top = draw_cuboid(ax, cursor, -h / 2, w, h, DECODER_FACE)
    label(ax, cursor + w / 2, top + 0.35, "fc + broadcast\n+ PositionalEnc", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, f"[T, {cfg['hidden_dim']}]")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # Decoder CircularResBlocks x n_layers (reverse dilation)
    w, h = 1.1, 3.0
    right, top = draw_stack(ax, cursor, -h / 2, w, h, DECODER_FACE, n=n)
    dils_rev = ", ".join(str(2 ** i) for i in reversed(range(n)))
    label(ax, cursor + 1.1, top + 0.35, f"CircularResBlock ×{n}\ndilation {dils_rev}", fontweight="bold")
    label(ax, cursor + 1.1, -h / 2 - 0.35, f"[T, {cfg['hidden_dim']}]")
    cursor = right + gap
    arrow(ax, cursor - gap, cursor)

    # final_projection -> output [T, 3]
    w, h = 1.8, 0.7
    right, top = draw_cuboid(ax, cursor, -h / 2, w, h, OUTPUT_FACE)
    label(ax, cursor + w / 2, top + 0.35, "final_projection\n+ activations", fontweight="bold")
    label(ax, cursor + w / 2, -h / 2 - 0.35, "[T, 3]\nspeed, steer, pos")
    cursor = right

    ax.set_xlim(-0.5, cursor + 0.5)
    ax.set_ylim(-3.2, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ── Full pipeline overview (vertical, half-page portrait) ─────────────────────

def build_full_pipeline_vertical(cfg):
    """Portrait (top-to-bottom) version of :func:`build_full_pipeline`.

    The same encoder → latent → decoder chain, drawn as a vertical stack of
    pseudo-3D slabs so it fits a tall half-page column. Each block keeps its
    name on the left and its tensor shape on the right; the latent bottleneck
    forks into ``fc_mu`` / ``fc_var`` and merges back through
    ``reparameterize``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 12))
    n = cfg["n_layers"]
    hid, lat, freq = cfg["hidden_dim"], cfg["latent_dim"], cfg["freq_bins"]

    W = 2.0            # default slab width
    arrow_len = 0.4    # visible length of the connecting arrows
    step = 0.15        # per-layer offset inside a CircularResBlock stack

    state = {"y": 0.0, "first": True}   # y = front-top of the next block (downward)

    def varrow(y0, y1, x=0.0):
        ax.add_patch(FancyArrowPatch(
            (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
            color=ARROW_COLOR, lw=2, zorder=4))

    def slab(title, shape, facecolor, h=0.5, accent=None, stack_n=None, w=W):
        """Drop one block below the previous one, wiring a downward arrow in."""
        overhang = DY + ((stack_n - 1) * step * (DY / DX) if stack_n else 0.0)
        if state["first"]:
            front_top = state["y"]
            state["first"] = False
        else:
            front_top = state["y"] - arrow_len - overhang
            varrow(state["y"], front_top + overhang)
        ybase = front_top - h
        if stack_n:
            right, _ = draw_stack(ax, -w / 2, ybase, w, h, facecolor, n=stack_n)
        else:
            right, _ = draw_cuboid(ax, -w / 2, ybase, w, h, facecolor)
            if accent:
                draw_cuboid(ax, w / 2 - 0.18, ybase, 0.18, h, accent)
        cy = ybase + h / 2 + overhang / 2
        label(ax, -w / 2 - 0.5, cy, title, ha="right", fontweight="bold", fontsize=9.5)
        label(ax, right + 0.4, cy, shape, ha="left", fontsize=9, color=SUBTLE)
        state["y"] = ybase
        return ybase

    dils = ", ".join(str(2 ** i) for i in range(n))
    dils_rev = ", ".join(str(2 ** i) for i in reversed(range(n)))

    # ── encoder ──
    slab("Input", "[T, 3]\nspeed, steer, pos", ENCODER_FACE, h=0.4, accent=ENCODER_ACCENT, w=1.5)
    slab("input_projection\nLinear(3→128)", f"[T, {hid}]", ENCODER_FACE)
    slab(f"CircularResBlock ×{n}\ndilation {dils}", f"[T, {hid}]", ENCODER_FACE, stack_n=n)
    slab(f"DFT Power Pool\nK={freq}", f"[{freq}, {hid}]", ENCODER_FACE, accent=ENCODER_ACCENT)

    # ── latent bottleneck: fc_mu / fc_var fork → reparameterize ──
    fork_y = state["y"]
    mh, mw = 0.55, 0.6
    mv_top = fork_y - arrow_len - DY
    mv_base = mv_top - mh
    cyl = mv_base + mh / 2 + DY / 2
    draw_cuboid(ax, -1.0, mv_base, mw, mh, LATENT_FACE)
    rv = draw_cuboid(ax, 0.4, mv_base, mw, mh, LATENT_FACE)[0]
    varrow(fork_y, mv_top + DY, x=-0.7)
    varrow(fork_y, mv_top + DY, x=0.7)
    label(ax, -1.0 - 0.35, cyl, f"fc_mu\nμ [{lat}]", ha="right", fontweight="bold", fontsize=9.5)
    label(ax, rv + 0.35, cyl, f"fc_var\nσ² [{lat}]", ha="left", fontweight="bold", fontsize=9.5)

    zh, zw = 0.6, 0.85
    z_top = mv_base - arrow_len - DY
    z_base = z_top - zh
    rz = draw_cuboid(ax, -zw / 2, z_base, zw, zh, LATENT_FACE)[0]
    dashed(ax, (-0.7, mv_base), (0, z_top + DY))
    dashed(ax, (0.7, mv_base), (0, z_top + DY))
    label(ax, rz + 0.35, z_base + zh / 2 + DY / 2, f"reparameterize\nz ~ N(μ, σ²)  [{lat}]",
          ha="left", fontweight="bold", fontsize=9.5)
    state["y"] = z_base

    # ── decoder ──
    slab("fc + broadcast\n+ PositionalEnc", f"[T, {hid}]", DECODER_FACE)
    slab(f"CircularResBlock ×{n}\ndilation {dils_rev}", f"[T, {hid}]", DECODER_FACE, stack_n=n)
    slab("final_projection\n+ activations", "[T, 3]\nspeed, steer, pos", OUTPUT_FACE, h=0.4)

    ax.set_xlim(-4.0, 3.9)
    ax.set_ylim(state["y"] - 0.7, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ── CircularResBlock internals ────────────────────────────────────────────────

def build_resblock_detail(cfg):
    fig, ax = plt.subplots(figsize=(16, 6))

    bw, bh = 2.4, 1.2
    gap = 0.7
    y = 0.0
    cursor = 0.0

    boxes = [
        ("x\n[B, C, T]", NORM_FACE),
        (f"CircularConv1d\nk={cfg['kernel_size']}, dilation=d", ENCODER_FACE),
        ("ChannelLayerNorm", NORM_FACE),
        ("GELU", LATENT_FACE),
        (f"CircularConv1d\nk={cfg['kernel_size']}, dilation=d", ENCODER_FACE),
        ("ChannelLayerNorm", NORM_FACE),
    ]

    centers = []
    for text, color in boxes:
        right, top = draw_box(ax, cursor, y, bw, bh, text, color)
        centers.append((cursor + bw / 2, cursor, right))
        if cursor > 0:
            arrow(ax, cursor - gap, cursor, y=y + bh / 2)
        cursor = right + gap

    # "+" residual-sum node
    r = 0.5
    circ = Circle((cursor + r, y + bh / 2), r, facecolor="white", edgecolor=EDGE_COLOR, lw=1.5, zorder=3)
    ax.add_patch(circ)
    label(ax, cursor + r, y + bh / 2, "+", fontsize=20, fontweight="bold")
    arrow(ax, cursor - gap, cursor, y=y + bh / 2)
    cursor = cursor + 2 * r + gap

    # output
    right, top = draw_box(ax, cursor, y, bw, bh, "output\n[B, C, T]", NORM_FACE)
    arrow(ax, cursor - gap, cursor, y=y + bh / 2)

    # residual connection: arc from input box, over the top, to the "+" node
    x_in_mid = bw / 2
    x_plus = (cursor - gap - 2 * r - gap) + r  # center x of the "+" circle
    y_plus_top = y + bh / 2 + r
    arrow(
        ax, x_in_mid, x_plus, y=y + bh + 1.6,
        connectionstyle="arc3,rad=0.0", arrowstyle="-",
    )
    # draw as a manual path instead: vertical-horizontal-vertical residual line
    ax.plot([x_in_mid, x_in_mid], [y + bh, y + bh + 1.6], color=ARROW_COLOR, lw=2, zorder=4)
    ax.plot([x_in_mid, x_plus], [y + bh + 1.6, y + bh + 1.6], color=ARROW_COLOR, lw=2, zorder=4)
    ax.add_patch(FancyArrowPatch(
        (x_plus, y + bh + 1.6), (x_plus, y_plus_top + 0.05),
        arrowstyle="-|>", mutation_scale=18, color=ARROW_COLOR, lw=2, zorder=4,
    ))
    label(ax, (x_in_mid + x_plus) / 2, y + bh + 1.9, "residual connection", fontstyle="italic", fontsize=10)

    label(
        ax, cursor / 2, y - 1.3,
        "Padded positions are masked to zero after every conv/norm step when sequences\n"
        "in a batch have unequal lengths (src_key_padding_mask).",
        fontsize=9, fontstyle="italic",
    )

    ax.set_xlim(-0.5, cursor + bw + 0.5)
    ax.set_ylim(y - 2.0, y + bh + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


# ── Circular padding concept ──────────────────────────────────────────────────

def build_circular_padding(cfg):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1.6]})

    # ── Left panel: closed-loop lap topology ────────────────────────────
    ax = axes[0]
    T_ring = 12
    radius = 2.0
    for i in range(T_ring):
        theta = 2 * math.pi * i / T_ring + math.pi / 2
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        c = Circle((x, y), 0.32, facecolor=ENCODER_FACE, edgecolor=EDGE_COLOR, lw=1.3, zorder=3)
        ax.add_patch(c)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, zorder=4)
    # direction arrow along the ring
    theta0 = 2 * math.pi * 1.5 / T_ring + math.pi / 2
    theta1 = 2 * math.pi * 2.5 / T_ring + math.pi / 2
    p0 = (radius * 1.35 * math.cos(theta0), radius * 1.35 * math.sin(theta0))
    p1 = (radius * 1.35 * math.cos(theta1), radius * 1.35 * math.sin(theta1))
    arrow(ax, p0[0], p1[0], y=p0[1])
    ax.text(0, -radius - 0.8, "Each track-metric sequence is one closed lap:\nstep T-1 is adjacent to step 0", ha="center", fontsize=10)

    ax.set_xlim(-radius - 1.2, radius + 1.2)
    ax.set_ylim(-radius - 1.8, radius + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Right panel: wrap-around padding before Conv1d ──────────────────
    ax = axes[1]
    T = 8
    pad = (cfg["kernel_size"] - 1) * 1  # dilation = 1 for illustration
    cell = 0.9

    # padding cells: copies of the last `pad` original positions
    for i in range(pad):
        idx = T - pad + i
        x = i * cell
        draw_box(ax, x, 0, cell * 0.95, cell, str(idx), shade(ENCODER_FACE, 1.25), fontsize=10)
        ax.add_patch(Rectangle((x, 0), cell * 0.95, cell, fill=False, edgecolor=EDGE_COLOR, lw=1.3, ls="--", zorder=4))

    # original sequence cells
    for i in range(T):
        x = (pad + i) * cell
        draw_box(ax, x, 0, cell * 0.95, cell, str(i), ENCODER_FACE, fontsize=10)

    total_w = (pad + T) * cell

    # bracket + arrow showing the wrap from the tail to the padding region
    tail_mid = (pad + (T - pad) + (T - 1)) / 2 * cell + cell / 2  # center of tail cells
    pad_mid = (pad / 2) * cell
    arrow(ax, tail_mid, pad_mid, y=cell + 1.0, connectionstyle="arc3,rad=-0.3")
    label(ax, (tail_mid + pad_mid) / 2, cell + 0.65, "copy last pad=(k-1)·dilation steps\n(wrap-around)", fontsize=9, fontstyle="italic")

    # kernel window over the first kernel_size cells
    k = cfg["kernel_size"]
    win = Rectangle((0, -0.15), k * cell, cell + 0.3, fill=False, edgecolor=ENCODER_ACCENT, lw=2.5, zorder=5)
    ax.add_patch(win)
    ax.add_patch(FancyArrowPatch(
        (k * cell / 2, -0.15), (k * cell / 2, -1.1),
        arrowstyle="-|>", mutation_scale=18, color=ENCODER_ACCENT, lw=2, zorder=5,
    ))
    label(ax, k * cell / 2, -1.45, f"Conv1d kernel (k={k})\nslides with no extra padding\n→ output[0]", fontsize=9, color=ENCODER_ACCENT)

    label(ax, total_w / 2, cell + 1.85, f"padding ({pad} cells)  +  original sequence (T={T} cells)", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-2.2, cell + 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig


# ── Dilation stack / receptive field ─────────────────────────────────────────

def build_dilation_field(cfg):
    """Stacked-layer "dilated convolutions" diagram (WaveNet-style): one row of
    cells per dilation level, with lines tracing the receptive field of a single
    output cell back down to the input. Norms/GELU/residuals are ignored —
    only the dilated CircularConv1d connectivity is shown."""
    n = cfg["n_layers"]
    dilations = [2 ** i for i in range(n)]  # encoder order: 1, 2, 4, 8, 16
    k = cfg["kernel_size"]  # real kernel size (7) — taps i, i-d, i-2d, ..., i-(k-1)d

    # Only a window of W cells is drawn per row; with k=7 the true receptive
    # field extends far to the left of the target, so an "..." cell stands in
    # for "earlier positions, off-screen" (and stays highlighted if part of
    # the receptive field still falls outside the window).
    W = 27
    target = W - 1

    # Trace the receptive field of `target` backwards through the stack.
    # CircularConv1d is causal: output[i] depends on input[i - tap*d] for tap in 0..k-1.
    active = {n: {target}}
    for l in range(n, 0, -1):
        d = dilations[l - 1]
        prev = set()
        for i in active[l]:
            for tap in range(k):
                prev.add(i - tap * d)
        active[l - 1] = prev

    cell_w, cell_h, row_gap = 1.0, 0.8, 2.625
    ellipsis_x = -1.4  # x-position of the "earlier positions" cell

    fig, ax = plt.subplots(figsize=(22, 2 + (n + 1) * row_gap))

    row_y = {l: l * row_gap for l in range(n + 1)}
    active_color = ENCODER_ACCENT
    inactive_colors = [NORM_FACE] + [ENCODER_FACE] * n  # row 0 = input, rows 1..n = conv outputs

    # edges (behind cells): solid for the "direct" tap (offset 0), dashed for dilated taps.
    # Only the fan-out of one representative cell per row is drawn (the one that
    # propagates `target` down via the tap=0 connection) — every other active cell
    # has an identical fan-out pattern, just shifted, so drawing all of them would
    # only add clutter.
    for l in range(n, 0, -1):
        d = dilations[l - 1]
        y0, y1 = row_y[l], row_y[l - 1]
        i = target
        for tap in range(k):
            j = i - tap * d
            style = "-" if tap == 0 else "--"
            x0 = i + cell_w / 2
            x1 = (ellipsis_x + cell_w * 0.6) if j < 0 else (j + cell_w / 2)
            ax.plot([x0, x1], [y0, y1 + cell_h], style, color=EDGE_COLOR, lw=1.2, alpha=0.8, zorder=2)

    # cells
    for l in range(n + 1):
        # "..." cell: highlighted if the receptive field extends past the window
        off_window_active = any(i < 0 for i in active[l])
        color = active_color if off_window_active else inactive_colors[l]
        ax.add_patch(Rectangle((ellipsis_x, row_y[l]), cell_w * 1.1, cell_h, facecolor=color, edgecolor=EDGE_COLOR, lw=1.0, zorder=3))
        label(ax, ellipsis_x + cell_w * 1.1 / 2, row_y[l] + cell_h / 2, "...", fontweight="bold")

        for i in range(W):
            color = active_color if i in active[l] else inactive_colors[l]
            ax.add_patch(Rectangle((i, row_y[l]), cell_w * 0.94, cell_h, facecolor=color, edgecolor=EDGE_COLOR, lw=1.0, zorder=3))

    # row labels
    row_labels = ["Input"] + [f"d={d}" for d in dilations]
    for l in range(n + 1):
        ax.text(ellipsis_x - 0.5, row_y[l] + cell_h / 2, row_labels[l], ha="right", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(ellipsis_x - 2.0, W + 0.5)
    ax.set_ylim(-0.5, n * row_gap + cell_h + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def main():
    cfg = MODEL_CONFIG

    save_fig(build_qd_pipeline(cfg), "qd_pipeline.png")
    save_fig(build_qd_pipeline_vertical(cfg), "qd_pipeline_vertical.png")
    save_fig(build_full_pipeline(cfg), "vae_architecture.png")
    save_fig(build_full_pipeline_vertical(cfg), "vae_architecture_vertical.png")
    save_fig(build_resblock_detail(cfg), "circular_resblock.png")
    save_fig(build_circular_padding(cfg), "circular_padding.png")
    save_fig(build_dilation_field(cfg), "dilation_receptive_field.png")


if __name__ == "__main__":
    main()
