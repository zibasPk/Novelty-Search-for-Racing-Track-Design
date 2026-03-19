"""
Interactive Plotly Dash app for visualizing Perlin noise track generation.

Run:
    python plot_perlin_experiments.py

Then open http://localhost:8050 in a browser.
Requires the track generation API running on localhost:4242.
"""

import requests
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, ctx
from dash.exceptions import PreventUpdate
import time

API_URL = "http://localhost:4242/genforweb"
MODE = "voronoi"
RNG_MODE_PERLIN = 1

# Global cache to allow instantaneous single-track enlargement 
# without needing to pass huge JSON strings back and forth through the browser.
TRACK_CACHE = {}
LAST_DATA_LIST = []
LAST_GENERATION_LABEL = ""

def generate_track(seed, perlin_params, track_size=None):
    if track_size is None:
        rng = np.random.RandomState(seed)
        track_size = int(rng.randint(4, 11))
    resp = requests.post(API_URL, json={
        "id": str(seed),
        "mode": MODE,
        "trackSize": track_size,
        "perlin_parameters": perlin_params,
        "rngMode": RNG_MODE_PERLIN,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_figure(data_list, cols=6):
    n = len(data_list)
    # Dynamically adjust columns to minimize empty space
    if n < cols:
        cols = n
    rows = max(1, math.ceil(n / cols))
    
    # We remove standard subplot_titles so we can add them as clickable plot traces
    fig = make_subplots(
        rows=rows, cols=cols,
        horizontal_spacing=0.006,
        vertical_spacing=0.02,
    )

    for idx, (seed, data) in enumerate(data_list):
        r = idx // cols + 1
        c = idx % cols + 1

        if data is None:
            fig.add_trace(go.Scatter(
                x=[300], y=[300], mode="text", text=["ERR"], 
                textfont=dict(size=18, color="red"), hoverinfo="skip", showlegend=False
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=[300], y=[30], mode="text", text=[f"#{seed}"],
                textfont=dict(size=14, color="#2196F3", weight="bold"),
                hovertext="Error generating track", hoverinfo="text",
                customdata=[seed], showlegend=False
            ), row=r, col=c)
            continue

        gen = data["generator"]
        track = data["track"]

        # 1. Clickable Subplot Title (The Name)
        fig.add_trace(go.Scatter(
            x=[300], y=[30], mode="text",
            text=[f"#{seed}"], textfont=dict(size=14, color="#2196F3", weight="bold"),
            hovertext="Click to enlarge", hoverinfo="text",
            customdata=[seed], showlegend=False
        ), row=r, col=c)

        # 2. Voronoi edges
        ex, ey = [], []
        for e in gen["diagram"]["edges"]:
            ex += [e["va"]["x"], e["vb"]["x"], None]
            ey += [e["va"]["y"], e["vb"]["y"], None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="#ddd", width=0.5),
            hoverinfo="skip", showlegend=False,
            customdata=[seed]*len(ex) if ex else []
        ), row=r, col=c)

        # 3. Points
        ds = gen["dataSet"]
        px, py = [p["x"] for p in ds], [p["y"] for p in ds]
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="markers", marker=dict(size=2, color="steelblue", opacity=0.6),
            hoverinfo="skip", showlegend=False,
            customdata=[seed]*len(px) if px else []
        ), row=r, col=c)

        # 4. Selected cells
        for cell in gen["selectedCells"]:
            site = cell["site"]
            pts = []
            for he in cell["halfedges"]:
                edge = he["edge"]
                ls = edge.get("lSite")
                if ls and ls["x"] == site["x"] and ls["y"] == site["y"]:
                    pts.append((edge["va"]["x"], edge["va"]["y"]))
                else:
                    pts.append((edge["vb"]["x"], edge["vb"]["y"]))
            if pts:
                cx = [p[0] for p in pts] + [pts[0][0]]
                cy = [p[1] for p in pts] + [pts[0][1]]
                fig.add_trace(go.Scatter(
                    x=cx, y=cy,
                    fill="toself", fillcolor="rgba(255,165,0,0.12)",
                    line=dict(width=0), mode="lines",
                    hoverinfo="skip", showlegend=False,
                    customdata=[seed]*len(cx)
                ), row=r, col=c)

        # 5. Track spline
        tx = [p["x"] for p in track] + [track[0]["x"]]
        ty = [p["y"] for p in track] + [track[0]["y"]]
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines",
            line=dict(color="crimson", width=1.5),
            hovertext="Click to enlarge", hoverinfo="text", showlegend=False,
            customdata=[seed]*len(tx)
        ), row=r, col=c)

    bbox = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}
    for i in range(1, n + 1):
        xax = f"xaxis{i}" if i > 1 else "xaxis"
        yax = f"yaxis{i}" if i > 1 else "yaxis"
        xref = f"x{i}" if i > 1 else "x"
        fig.layout[xax].update(
            range=[bbox["xl"], bbox["xr"]],
            showticklabels=False, showgrid=False, zeroline=False,
            constrain="domain",
        )
        fig.layout[yax].update(
            range=[bbox["yb"], bbox["yt"]],
            showticklabels=False, showgrid=False, zeroline=False,
            scaleanchor=xref,
            scaleratio=1,
            constrain="domain"
        )

    fig.update_layout(
        height=max(1, rows) * 300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        clickmode="event"
    )
    return fig


def build_single_figure(seed, data):
    """Builds a large, detailed figure for a single track inside the modal."""
    fig = go.Figure()
    gen = data["generator"]
    track = data["track"]

    # Voronoi edges
    ex, ey = [], []
    for e in gen["diagram"]["edges"]:
        ex += [e["va"]["x"], e["vb"]["x"], None]
        ey += [e["va"]["y"], e["vb"]["y"], None]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines", line=dict(color="#ddd", width=1.0),
        hoverinfo="skip", showlegend=False
    ))

    # Points
    ds = gen["dataSet"]
    fig.add_trace(go.Scatter(
        x=[p["x"] for p in ds], y=[p["y"] for p in ds], mode="markers", 
        marker=dict(size=4, color="steelblue", opacity=0.8), hoverinfo="skip", showlegend=False
    ))

    # Selected cells
    for cell in gen["selectedCells"]:
        site = cell["site"]
        pts = []
        for he in cell["halfedges"]:
            edge = he["edge"]
            ls = edge.get("lSite")
            if ls and ls["x"] == site["x"] and ls["y"] == site["y"]:
                pts.append((edge["va"]["x"], edge["va"]["y"]))
            else:
                pts.append((edge["vb"]["x"], edge["vb"]["y"]))
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts] + [pts[0][0]], y=[p[1] for p in pts] + [pts[0][1]],
                fill="toself", fillcolor="rgba(255,165,0,0.2)", line=dict(width=0), 
                mode="lines", hoverinfo="skip", showlegend=False
            ))

    # Track spline
    tx = [p["x"] for p in track] + [track[0]["x"]]
    ty = [p["y"] for p in track] + [track[0]["y"]]
    fig.add_trace(go.Scatter(
        x=tx, y=ty, mode="lines", line=dict(color="crimson", width=3),
        hoverinfo="skip", showlegend=False
    ))

    bbox = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}
    fig.update_layout(
        xaxis=dict(range=[bbox["xl"], bbox["xr"]], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[bbox["yb"], bbox["yt"]], showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
        margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="white", plot_bgcolor="white", height=600
    )
    return fig


def render_tracks_matplotlib_png(data_list, title_text, cols=6):
    n = len(data_list)
    if n == 0:
        raise ValueError("No tracks to render")

    if n < cols:
        cols = n
    rows = max(1, math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1), dpi=180)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        if idx >= n:
            ax.axis("off")
            continue

        seed, data = data_list[idx]
        ax.set_xlim(0, 600)
        ax.set_ylim(600, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if data is None:
            ax.text(300, 30, f"#{seed}", ha="center", va="center", color="#2196F3", fontsize=9, fontweight="bold")
            ax.text(300, 300, "ERR", ha="center", va="center", color="red", fontsize=13)
            continue

        gen = data["generator"]
        track = data["track"]

        ax.text(300, 30, f"#{seed}", ha="center", va="center", color="#2196F3", fontsize=9, fontweight="bold")

        ex, ey = [], []
        for e in gen["diagram"]["edges"]:
            ex += [e["va"]["x"], e["vb"]["x"], np.nan]
            ey += [e["va"]["y"], e["vb"]["y"], np.nan]
        if ex:
            ax.plot(ex, ey, color="#dddddd", linewidth=0.5)

        ds = gen["dataSet"]
        if ds:
            ax.scatter([p["x"] for p in ds], [p["y"] for p in ds], s=4, c="steelblue", alpha=0.6, linewidths=0)

        for cell in gen["selectedCells"]:
            site = cell["site"]
            pts = []
            for he in cell["halfedges"]:
                edge = he["edge"]
                ls = edge.get("lSite")
                if ls and ls["x"] == site["x"] and ls["y"] == site["y"]:
                    pts.append((edge["va"]["x"], edge["va"]["y"]))
                else:
                    pts.append((edge["vb"]["x"], edge["vb"]["y"]))
            if pts:
                poly = plt.Polygon(pts, closed=True, facecolor=(1.0, 0.647, 0.0, 0.12), edgecolor="none")
                ax.add_patch(poly)

        tx = [p["x"] for p in track] + [track[0]["x"]]
        ty = [p["y"] for p in track] + [track[0]["y"]]
        ax.plot(tx, ty, color="crimson", linewidth=1.5)

    if title_text:
        fig.suptitle(title_text, fontsize=10)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93 if title_text else 0.99, wspace=0.06, hspace=0.12)
    return fig


# ── App ───────────────────────────────────────────────────────

# Ensure prevent_initial_callbacks ignores any lingering dynamic errors
app = Dash(__name__, title="Perlin Track Explorer", suppress_callback_exceptions=True)

label_style = {"fontSize": "0.85rem", "fontWeight": "500", "marginBottom": "2px"}
input_style = {"width": "100%"}
INFO_BTN_CSS = {
    "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
    "width": "16px", "height": "16px", "borderRadius": "50%",
    "backgroundColor": "#2196F3", "color": "white", "fontSize": "11px",
    "fontWeight": "bold", "cursor": "default", "marginLeft": "4px",
    "lineHeight": "1", "flexShrink": "0", "userSelect": "none",
}

def param_label(name, tooltip):
    return html.Div([
        html.Span(name, style=label_style),
        html.Span("i", title=tooltip, style=INFO_BTN_CSS),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "2px"})

app.layout = html.Div([
    html.H3("Perlin Track Explorer", style={"margin": "12px 16px 4px"}),

    html.Div([
        html.Div([
            param_label("Frequency", "Perlin noise frequency (NOISE_FREQUENCY). Higher values produce more fine-grained noise, creating denser and more complex track layouts. Range: 1–20."),
            dcc.Input(id="feat", type="number", value=3, min=1, max=20, step=0.1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Threshold", "Density threshold (densityThreshold). Sets the cutoff for cell selection based on Perlin noise. Range: 0–0.9."),
            dcc.Input(id="bias", type="number", value=0.3, min=0, max=0.9, step=0.05, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Density exponent", "Density exponent (densityExponent). Shapes the spatial distribution of sites. Range: 0.1–6."),
            dcc.Input(id="power", type="number", value=2.0, min=0.1, max=6, step=0.1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Track Size", "Number of Voronoi sites used for track generation. 0 = randomly chosen between 4 and 10. Range: 0–15."),
            dcc.Input(id="tsize", type="number", value=0, min=0, max=15, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Seeds", "Number of random seeds to generate tracks for in a single batch. Range: 1–200."),
            dcc.Input(id="num-seeds", type="number", value=30, min=1, max=200, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Seed Offset", "Starting seed number. Seeds will range from offset to offset + num_seeds − 1."),
            dcc.Input(id="seed-offset", type="number", value=0, min=0, max=10000, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Min Dist Scale", "Minimum distance scale (minDistScale). Controls how close Voronoi sites can be. Default: 0.25."),
            dcc.Input(id="min-dist-scale", type="number", value=0.25, min=0.05, max=2, step=0.01, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            html.Br(),
            html.Button("Generate", id="btn-gen", n_clicks=0,
                         style={"padding": "8px 24px", "fontSize": "1rem",
                                "cursor": "pointer", "backgroundColor": "#2196F3",
                                "color": "white", "border": "none", "borderRadius": "4px"}),
        ], style={"padding": "0 8px", "display": "flex", "alignItems": "end"}),

        html.Div([
            html.Br(),
            html.Button("Save PNG", id="btn-save-png", n_clicks=0,
                         style={"padding": "8px 24px", "fontSize": "1rem",
                                "cursor": "pointer", "backgroundColor": "#4CAF50",
                                "color": "white", "border": "none", "borderRadius": "4px"}),
        ], style={"padding": "0 8px", "display": "flex", "alignItems": "end"}),
    ], style={"display": "flex", "padding": "8px", "alignItems": "end", "flexWrap": "wrap", "gap": "4px"}),

    dcc.Download(id="download-tracks-png"),

    # We now place the main-graph directly in the layout, hidden until generated.
    dcc.Loading(
        dcc.Graph(
            id="main-graph", 
            style={"display": "none"}, # Hidden initially
            config={"scrollZoom": False, "displayModeBar": True,
                    "toImageButtonOptions": {"format": "png", "scale": 2}}
        ),
        type="circle",
    ),

    # ── Modal Overlay for Enqueue Plot ──
    html.Div(id="modal-container", style={"display": "none"}, children=[
        html.Div(style={
            "position": "fixed", "top": 0, "left": 0, "width": "100%", "height": "100%",
            "backgroundColor": "rgba(0,0,0,0.6)", "zIndex": 1000,
            "display": "flex", "justifyContent": "center", "alignItems": "center"
        }, children=[
            html.Div(style={
                "backgroundColor": "white", "padding": "20px", "borderRadius": "8px",
                "width": "90%", "maxWidth": "700px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)",
                "position": "relative"
            }, children=[
                html.Button("✖", id="btn-close-modal", style={
                    "position": "absolute", "top": "15px", "right": "15px", "zIndex": 1001,
                    "background": "none", "border": "none", "fontSize": "24px", "cursor": "pointer"
                }),
                html.H3(id="modal-title", style={"marginTop": 0, "marginBottom": "10px", "color": "#2196F3"}),
                dcc.Graph(id="modal-graph", config={"displayModeBar": True})
            ])
        ])
    ])
], style={"fontFamily": "system-ui, sans-serif"})


@callback(
    Output("main-graph", "figure"),
    Output("main-graph", "style"),
    Input("btn-gen", "n_clicks"),
    State("feat", "value"), State("bias", "value"),
    State("power", "value"), State("tsize", "value"),
    State("num-seeds", "value"), State("seed-offset", "value"),
    State("min-dist-scale", "value"),
    prevent_initial_call=True,
)
def on_generate(n, feat, bias, power, tsize, num_seeds, offset, min_dist_scale):
    if not n:
        return no_update, no_update

    params = {
        "NOISE_FREQUENCY": float(feat or 3),
        "densityThreshold": float(bias if bias is not None else 0.3),
        "densityExponent": float(power or 2.0),
        "minDistScale": float(min_dist_scale or 0.25),
    }
    tsize = int(tsize or 0) or None
    num_seeds = int(num_seeds or 30)
    offset = int(offset or 0)
    seeds = list(range(offset, offset + num_seeds))

    global TRACK_CACHE, LAST_DATA_LIST, LAST_GENERATION_LABEL
    TRACK_CACHE.clear()

    data_list, ok = [], 0
    t0 = time.time()
    for s in seeds:
        try:
            d = generate_track(s, params, track_size=tsize)
            data_list.append((s, d))
            TRACK_CACHE[s] = d
            ok += 1
        except Exception:
            data_list.append((s, None))
    elapsed = time.time() - t0

    fig = build_figure(data_list)
    size_label = "random(4-10)" if tsize is None else str(tsize)
    label = (
        f"freq={params['NOISE_FREQUENCY']:.2f}  "
        f"threshold={params['densityThreshold']:.2f}  "
        f"exp={params['densityExponent']:.2f}  "
        f"minDist={params['minDistScale']:.2f}  "
        f"trackSize={size_label}  "
        f"seeds={offset}-{offset + num_seeds - 1}"
    )
    LAST_DATA_LIST = data_list
    LAST_GENERATION_LABEL = f"{label}   ({ok}/{num_seeds} ok, {elapsed:.1f}s)"
    fig.update_layout(title=dict(
        text=f"{label}   ({ok}/{num_seeds} ok, {elapsed:.1f}s) — <b>Click on a track or its name to enlarge</b>",
        font=dict(size=13),
    ))

    # Return the figure AND a style dict to un-hide the graph on the frontend
    return fig, {"width": "100%", "display": "block"}


@callback(
    Output("download-tracks-png", "data"),
    Input("btn-save-png", "n_clicks"),
    prevent_initial_call=True,
)
def save_tracks_png_matplotlib(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    if not LAST_DATA_LIST:
        raise PreventUpdate

    def _writer(bytes_io):
        fig = render_tracks_matplotlib_png(LAST_DATA_LIST, LAST_GENERATION_LABEL, cols=6)
        fig.savefig(bytes_io, format="png", dpi=180, facecolor="white")
        plt.close(fig)

    return dcc.send_bytes(_writer, "perlin_tracks_matplotlib.png")


@callback(
    Output("modal-container", "style"),
    Output("modal-graph", "figure"),
    Output("modal-title", "children"),
    Input("main-graph", "clickData"),
    Input("btn-close-modal", "n_clicks"),
    prevent_initial_call=True
)
def handle_click(click_data, close_clicks):
    """Handles opening a large single-track plot when clicking inside the main grid."""
    trigger = ctx.triggered_id

    # Close button clicked
    if trigger == "btn-close-modal":
        return {"display": "none"}, go.Figure(), ""

    # Graph clicked
    if trigger == "main-graph" and click_data:
        points = click_data.get("points", [])
        if not points:
            return no_update

        # Find the seed based on clicked trace customdata
        seed = points[0].get("customdata")
        if seed is None:
            return no_update
        if isinstance(seed, list):
            seed = seed[0]

        data = TRACK_CACHE.get(seed)
        if data is None:
            # Handle error tracks
            fig = go.Figure()
            fig.add_annotation(text="ERR or No Data", x=300, y=300, showarrow=False, font=dict(size=20, color="red"))
            fig.update_layout(xaxis=dict(range=[0,600], showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(range=[600,0], showgrid=False, zeroline=False, showticklabels=False))
        else:
            fig = build_single_figure(seed, data)

        return {"display": "block"}, fig, f"Track #{seed}"

    return no_update


if __name__ == "__main__":
    app.run(debug=True, port=8050)