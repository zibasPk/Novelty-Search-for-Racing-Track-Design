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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State, no_update
import time

API_URL = "http://localhost:4242/genforweb"
MODE = "voronoi"


def generate_track(seed, perlin_params, track_size=None):
    if track_size is None:
        rng = np.random.RandomState(seed)
        track_size = int(rng.randint(4, 11))
    resp = requests.post(API_URL, json={
        "id": str(seed),
        "mode": MODE,
        "trackSize": track_size,
        "perlin_parameters": perlin_params,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_figure(data_list, titles, cols=6):
    n = len(data_list)
    rows = max(1, math.ceil(n / cols))
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )

    for idx, (seed, data) in enumerate(data_list):
        r = idx // cols + 1
        c = idx % cols + 1

        if data is None:
            fig.add_annotation(
                text="ERR", xref=f"x{idx+1}", yref=f"y{idx+1}",
                x=300, y=300, showarrow=False,
                font=dict(size=14, color="red"), row=r, col=c,
            )
            continue

        gen = data["generator"]
        track = data["track"]

        # Voronoi edges
        ex, ey = [], []
        for e in gen["diagram"]["edges"]:
            ex += [e["va"]["x"], e["vb"]["x"], None]
            ey += [e["va"]["y"], e["vb"]["y"], None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="#ddd", width=0.5),
            hoverinfo="skip", showlegend=False,
        ), row=r, col=c)

        # Points
        ds = gen["dataSet"]
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in ds], y=[p["y"] for p in ds],
            mode="markers", marker=dict(size=2, color="steelblue", opacity=0.6),
            hoverinfo="skip", showlegend=False,
        ), row=r, col=c)

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
                    x=[p[0] for p in pts] + [pts[0][0]],
                    y=[p[1] for p in pts] + [pts[0][1]],
                    fill="toself", fillcolor="rgba(255,165,0,0.12)",
                    line=dict(width=0), mode="lines",
                    hoverinfo="skip", showlegend=False,
                ), row=r, col=c)

        # Track spline
        tx = [p["x"] for p in track] + [track[0]["x"]]
        ty = [p["y"] for p in track] + [track[0]["y"]]
        fig.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines",
            line=dict(color="crimson", width=1.5),
            hoverinfo="skip", showlegend=False,
        ), row=r, col=c)

    bbox = {"xl": 0, "xr": 600, "yt": 0, "yb": 600}
    for i in range(1, n + 1):
        xax = f"xaxis{i}" if i > 1 else "xaxis"
        yax = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xax].update(
            range=[bbox["xl"], bbox["xr"]],
            showticklabels=False, showgrid=False, zeroline=False,
            scaleanchor=yax.replace("axis", ""),
        )
        fig.layout[yax].update(
            range=[bbox["yb"], bbox["yt"]],
            showticklabels=False, showgrid=False, zeroline=False,
        )

    fig.update_layout(
        height=max(1, rows) * 250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10
    return fig


# ── App ───────────────────────────────────────────────────────

app = Dash(__name__, title="Perlin Track Explorer")

label_style = {"fontSize": "0.85rem", "fontWeight": "500", "marginBottom": "2px"}
input_style = {"width": "100%"}
tip_style = {"fontSize": "0.7rem", "color": "#888", "marginTop": "2px", "lineHeight": "1.2"}

INFO_BTN_CSS = {
    "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
    "width": "16px", "height": "16px", "borderRadius": "50%",
    "backgroundColor": "#2196F3", "color": "white", "fontSize": "11px",
    "fontWeight": "bold", "cursor": "default", "marginLeft": "4px",
    "lineHeight": "1", "flexShrink": "0", "userSelect": "none",
}

def param_label(name, tooltip):
    """Return a label row with an info icon that shows a tooltip on hover."""
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
                param_label("Threshold", "Density threshold (densityThreshold). Sets the cutoff for cell selection based on Perlin noise. Lower values select more cells, leading to denser spatial coverage and more uniform distribution. Higher values select fewer cells, concentrating sites in high-noise regions and creating sparser clusters. Range: 0–0.9."),
            dcc.Input(id="bias", type="number", value=0.3, min=0, max=0.9, step=0.05, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
                param_label("Density exponent", "Density exponent (densityExponent). Shapes the spatial distribution of sites: Values > 1 amplify the effect of the threshold, clustering sites tightly in high-noise regions (sparser, more selective). Values < 1 flatten the noise, spreading sites more evenly and increasing coverage. At 1, noise is used as-is. Range: 0.1–6."),
            dcc.Input(id="power", type="number", value=2.0, min=0.1, max=6, step=0.1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Track Size", "Number of Voronoi sites used for track generation. 0 = randomly chosen between 4 and 10. Range: 0–15."),
            dcc.Input(id="tsize", type="number", value=0, min=0, max=15, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Seeds", "Number of random seeds to generate tracks for in a single batch. Each seed produces a unique track. Range: 1–200."),
            dcc.Input(id="num-seeds", type="number", value=30, min=1, max=200, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            param_label("Seed Offset", "Starting seed number. Seeds will range from offset to offset + num_seeds − 1. Useful for exploring different track populations."),
            dcc.Input(id="seed-offset", type="number", value=0, min=0, max=10000, step=1, style=input_style),
        ], style={"flex": "1", "padding": "0 8px"}),

            html.Div([
                param_label("Min Dist Scale", "Minimum distance scale (minDistScale). Controls how close Voronoi sites can be to each other. Lower values allow more sites (denser point cloud, more complex tracks), higher values enforce more spacing (sparser, simpler tracks). Typical range: 0.1–1.0. Default: 0.25."),
                dcc.Input(id="min-dist-scale", type="number", value=0.25, min=0.05, max=2, step=0.01, style=input_style),
            ], style={"flex": "1", "padding": "0 8px"}),

        html.Div([
            html.Br(),
            html.Button("Generate", id="btn-gen", n_clicks=0,
                         style={"padding": "8px 24px", "fontSize": "1rem",
                                "cursor": "pointer", "backgroundColor": "#2196F3",
                                "color": "white", "border": "none", "borderRadius": "4px"}),
        ], style={"padding": "0 8px", "display": "flex", "alignItems": "end"}),
    ], style={"display": "flex", "padding": "8px", "alignItems": "end",
              "flexWrap": "wrap", "gap": "4px"}),

    dcc.Loading(
        html.Div(id="output"),
        type="circle",
    ),
], style={"fontFamily": "system-ui, sans-serif"})


@callback(
    Output("output", "children"),
    Input("btn-gen", "n_clicks"),
    State("feat", "value"), State("bias", "value"),
    State("power", "value"), State("tsize", "value"),
    State("num-seeds", "value"), State("seed-offset", "value"),
    State("min-dist-scale", "value"),
    prevent_initial_call=True,
)
def on_generate(n, feat, bias, power, tsize, num_seeds, offset, min_dist_scale):

    if not n:
        return no_update



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

    data_list, titles, ok = [], [], 0
    t0 = time.time()
    for s in seeds:
        try:
            d = generate_track(s, params, track_size=tsize)
            data_list.append((s, d))
            titles.append(f"#{s}")
            ok += 1
        except Exception:
            data_list.append((s, None))
            titles.append(f"#{s} ERR")
    elapsed = time.time() - t0

    fig = build_figure(data_list, titles)
    label = f"feat={params['NOISE_FREQUENCY']:.0f}  bias={params['densityThreshold']:.2f}  pow={params['densityExponent']:.1f}"
    fig.update_layout(title=dict(
        text=f"{label}   ({ok}/{num_seeds} ok, {elapsed:.1f}s)",
        font=dict(size=13),
    ))

    return dcc.Graph(
        figure=fig,
        config={"scrollZoom": False, "displayModeBar": True,
                "toImageButtonOptions": {"format": "png", "scale": 2}},
        style={"width": "100%"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
