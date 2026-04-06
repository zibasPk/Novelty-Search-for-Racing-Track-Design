import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

def plot_track(ax, track, title=None, track_color="crimson"):
    tx = [point['x'] for point in track]
    ty = [point['y'] for point in track]
   
    
    ax.plot(tx, ty, color=track_color, linewidth=1.2, zorder=4)

    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if title: ax.set_title(title, fontsize=8, pad=4)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d


def _prepare_trace_for_interpolation(track_cum_dist, heatmap_data):
    """Return cleaned values/distances and align distance scale to the plotted track."""
    if not heatmap_data:
        return None, None

    arr = np.asarray(heatmap_data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, None

    values = arr[:, 0]
    distances = arr[:, 1]

    finite_mask = np.isfinite(values) & np.isfinite(distances)
    values = values[finite_mask]
    distances = distances[finite_mask]
    if len(values) < 2:
        return None, None

    # Ensure strictly non-decreasing distance for interpolation.
    order = np.argsort(distances, kind="stable")
    distances = distances[order]
    values = values[order]

    # Remove duplicate distances so interp1d does not get ambiguous x values.
    unique_distances, unique_idx = np.unique(distances, return_index=True)
    unique_values = values[unique_idx]
    if len(unique_values) < 2:
        return None, None

    track_len = float(track_cum_dist[-1]) if len(track_cum_dist) > 0 else 0.0
    trace_len = float(unique_distances[-1])
    if track_len > 0 and trace_len > 0:
        ratio = trace_len / track_len
        # Auto-align when traces are in a scaled distance domain (e.g., TORCS x2 units).
        if abs(ratio - 1.0) > 0.05:
            unique_distances = unique_distances / ratio

    return unique_values, unique_distances

def plot_track_heatmap(ax, track, heatmap_data, title=None, cmap="plasma", color_range=None):
    """
    Parameters
    ----------
    ax           : matplotlib Axes
    track        : list of {'x': ..., 'y': ...} dicts
    heatmap_data : list of (value, distance) pairs, where distance is from track start
    title        : optional title string
    cmap         : colormap name
    color_range  : optional (vmin, vmax) tuple
    """
    tx = np.array([p['x'] for p in track])
    ty = np.array([p['y'] for p in track])

    # --- compute cumulative distance along the track ---
    diffs = np.hypot(np.diff(tx), np.diff(ty))
    cum_dist = np.concatenate([[0], np.cumsum(diffs)])

    # --- interpolate heatmap values onto every track point ---
    values, distances = _prepare_trace_for_interpolation(cum_dist, heatmap_data)
    if values is None:
        return

    interp = interp1d(distances, values, bounds_error=False,
                      fill_value=(values[0], values[-1]))
    point_values = interp(cum_dist)

    # --- build a LineCollection: one segment per pair of consecutive points ---
    points = np.stack([tx, ty], axis=1)[:, None, :]   # (N, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)

    if color_range is None:
        color_range = (values.min(), values.max())
    norm = Normalize(vmin=color_range[0], vmax=color_range[1])
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=4)
    lc.set_array((point_values[:-1] + point_values[1:]) / 2)  # mid-segment value
    ax.add_collection(lc)

    ax.set_xlim(tx.min(), tx.max())
    ax.set_ylim(ty.min(), ty.max())
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    if title:
        ax.set_title(title, fontsize=8, pad=4)


def plot_trace_fixed_line(ax, trace_values, line_length=1.0, y=0.0, title=None,
                          cmap="plasma", color_range=None, linewidth=8,
                          show_colorbar=True):
    """Plot a 1D trace as a colored line with a fixed geometric length.

    Parameters
    ----------
    ax           : matplotlib Axes
    trace_values : array-like of shape (N,)
    line_length  : total length of the rendered line in axis units
    y            : y-position of the line
    title        : optional title string
    cmap         : colormap name
    color_range  : optional (vmin, vmax) tuple
    linewidth    : line thickness
    show_colorbar: whether to draw a colorbar
    """
    values = np.asarray(trace_values, dtype=float).ravel()
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    # Need at least two values to build one segment.
    if values.size < 2:
        return

    x = np.linspace(0.0, float(line_length), values.size)
    y_arr = np.full_like(x, float(y))

    points = np.stack([x, y_arr], axis=1)[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if color_range is None:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    else:
        vmin, vmax = color_range

    # Avoid zero-range normalization for constant traces.
    if vmin == vmax:
        eps = 1e-12
        vmin -= eps
        vmax += eps

    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    lc.set_array((values[:-1] + values[1:]) / 2.0)
    ax.add_collection(lc)

    ax.set_xlim(0.0, float(line_length))
    half_lw = max(0.05, linewidth * 0.01)
    ax.set_ylim(float(y) - half_lw, float(y) + half_lw)
    ax.set_aspect("auto")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if show_colorbar:
        plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
    if title:
        ax.set_title(title, fontsize=8, pad=4)
        
