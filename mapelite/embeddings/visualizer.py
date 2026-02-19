import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent
DATASETS_FOLDER = BASE_DIR / "datasets/embeddings/good"
TRACKS_FILE = BASE_DIR / "datasets/tracks.npz"
FITNESS_FILE = BASE_DIR / "datasets/fitness_dict.npz"

DEFAULT_MAX_POINTS = 9000

# Fixed ranges for consistent visualization
TRACE_RANGES = {
    "speed_trace": [0, 60],
    "steer_trace": [-1, 1],
    "accel_trace": [0, 1],
    "brake_trace": [0, 1]
}

# ==========================================
# 2. DATA LOADING & HELPERS
# ==========================================

if TRACKS_FILE.exists():
    raw_tracks_file = np.load(TRACKS_FILE, allow_pickle=True)
    tracks_dict = dict(raw_tracks_file)
    print(f"Loaded {len(tracks_dict)} tracks.")
else:
    tracks_dict = {}
    print(f"ERROR: {TRACKS_FILE} not found.")

fitness_data = None
scalar_metrics = []
trace_metrics = ["speed_trace", "accel_trace", "steer_trace", "brake_trace"]

if FITNESS_FILE.exists():
    try:
        fitness_data = np.load(FITNESS_FILE, allow_pickle=True)
        if len(fitness_data.files) > 0:
            first_id = fitness_data.files[0]
            first_entry = fitness_data[first_id].item()
            scalar_metrics = sorted([k for k, v in first_entry.items() if isinstance(v, (int, float))])
            print(f"Loaded fitness. Scalar metrics: {len(scalar_metrics)}")
    except Exception as e:
        print(f"Error loading fitness file: {e}")
        
        
custom_burd = [
    [0.0, 'rgb(33, 102, 172)'],   # Dark Blue (Min)
    [0.4, 'rgb(103, 169, 207)'],  # Light Blue
    [0.5, 'rgb(255, 255, 255)'],  # White (Center)
    [0.6, 'rgb(239, 138, 98)'],   # Light Red
    [1.0, 'rgb(178, 24, 43)']     # Dark Red (Max)
]

        
def interpolate_metrics_to_track(track_xy, trace_data):
    if not trace_data or len(trace_data) < 2: return None
    deltas = np.diff(track_xy, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    track_dist = np.insert(np.cumsum(seg_lengths), 0, 0)
    trace_arr = np.array(trace_data)
    t_values = trace_arr[:, 0]
    t_dists = trace_arr[:, 1]
    return np.interp(track_dist, t_dists, t_values)

def get_available_datasets():
    if not DATASETS_FOLDER.exists(): return []
    return sorted([f.name for f in DATASETS_FOLDER.glob("*.npz")])

def load_dataset(filename):
    filepath = DATASETS_FOLDER / filename
    if not filepath.exists(): return None, None
    data = np.load(filepath)
    return data["embeddings"], data["ids"]

def prepare_dataframe(latents, raw_ids, max_points, selected_metric, use_robust=True):
    # 1. Create Basic DataFrame
    df = pd.DataFrame(latents, columns=['Latent_X', 'Latent_Y'])
    df['ID'] = [str(x) for x in raw_ids]
    
    # 2. Extract Raw Metric Values
    if selected_metric and fitness_data and selected_metric != "None":
        values = []
        for uid in df['ID']:
            if uid in fitness_data:
                # fetch the value safely
                val = fitness_data[uid].item().get(selected_metric, np.nan)
                values.append(val)
            else:
                values.append(np.nan)
        
        # Store the REAL values for the Tooltip
        df[selected_metric] = values
        
        # 3. Create a SCALED column for the Color Map
        if use_robust:
            # Drop NaNs for the scaler training, then fill back or handle NaNs
            # Simple approach: Fill NaNs with median for scaling or ignore
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) > 0:
                scaler = RobustScaler()
                # scaler = StandardScaler()  # Alternative: StandardScaler for Z-score normalization
                # Reshape needed for sklearn (n_samples, n_features)
                raw_np = np.array(values)[valid_mask].reshape(-1, 1)
                scaled_np = scaler.fit_transform(raw_np)
                
                # Initialize scaled column with NaNs
                df[f"{selected_metric}_scaled"] = np.nan
                # Fill in the valid scaled values
                df.loc[valid_mask, f"{selected_metric}_scaled"] = scaled_np.flatten()
            else:
                df[f"{selected_metric}_scaled"] = df[selected_metric]
        else:
            # If not robust, just copy the data
            df[f"{selected_metric}_scaled"] = df[selected_metric]

    # 4. Downsampling (Optimization)
    if max_points and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
        
    return df

# ==========================================
# 3. LAYOUT
# ==========================================
datasets = get_available_datasets()
init_ds = datasets[0] if datasets else None

opt_ds = [{'label': d, 'value': d} for d in datasets]
opt_scalar = [{'label': "None (Blue)", 'value': "None"}] + [{'label': m, 'value': m} for m in scalar_metrics]
opt_trace = [{'label': "None (Solid Color)", 'value': "None"}] + \
            [{'label': t.replace('_trace', '').capitalize(), 'value': t} for t in trace_metrics]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Voronoi Evolution Explorer", style={'fontFamily': 'sans-serif', 'textAlign': 'center', 'color': '#333'}),
    
    html.Div([
        html.Div([
            html.Label("Dataset:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='dataset-dropdown', options=opt_ds, value=init_ds, clearable=False),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '5px'}),
        
        html.Div([
            html.Label("Color Latent By:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='color-metric-dropdown', options=opt_scalar, value="None", clearable=False),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '5px'}),

        html.Div([
            html.Label("Max Samples:", style={'fontWeight': 'bold'}),
            dcc.Input(id='sample-count', type='number', value=DEFAULT_MAX_POINTS, step=500),
        ], style={'width': '10%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Search ID:", style={'fontWeight': 'bold'}),
            dcc.Input(id='search-input', type='text', placeholder='ID...', debounce=True),
        ], style={'width': '15%', 'display': 'inline-block', 'padding': '5px', 'verticalAlign': 'top'}),
    ], style={'backgroundColor': '#f4f4f4', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='latent-graph', style={'height': '85vh'})
        ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.Div([
                html.Label("Overlay Metric:", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                dcc.Dropdown(id='track-metric-dropdown', options=opt_trace, value="speed_trace", clearable=False, style={'width': '50%', 'display': 'inline-block'}),
            ], style={'padding': '5px', 'backgroundColor': '#f9f9f9', 'borderBottom': '1px solid #ccc'}),
            
            dcc.Graph(id='track-graph', style={'height': '40vh'}),
            dcc.Graph(id='trace-histogram', style={'height': '20vh', 'marginTop': '5px'}),
            
            html.H4("Fitness Details", style={'margin': '10px 0 5px 0', 'borderBottom': '1px solid #ccc'}),
            html.Div(id='fitness-table-container', style={'height': '20vh', 'overflowY': 'auto', 'padding': '5px', 'backgroundColor': '#222', 'color': 'white', 'fontSize': '0.9em'})
            
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'borderLeft': '1px solid #ccc', 'paddingLeft': '10px'})
    ], style={'display': 'flex', 'marginTop': '0px'})
])

# ==========================================
# 4. CALLBACKS
# ==========================================

@app.callback(
    Output('latent-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('sample-count', 'value'),
     Input('search-input', 'value'),
     Input('color-metric-dropdown', 'value')]
)
def update_latent(dataset, max_samples, search_id, color_col):
    if not dataset: return go.Figure()
    latents, raw_ids = load_dataset(dataset)
    if latents is None: return go.Figure()

    # Pass use_robust=True to generate the extra column
    df = prepare_dataframe(latents, raw_ids, max_samples, color_col, use_robust=True)
    
    highlight_id = str(search_id).strip() if search_id else None
    title = f"Latent Space: {dataset}"

    if color_col and color_col != "None" and color_col in df.columns:
        # DETERMINE WHICH COLUMN TO USE FOR COLOR
        color_source = f"{color_col}_scaled"
        
        fig = px.scatter(
            df, 
            x='Latent_X', 
            y='Latent_Y', 
            
            # Use the Scaled column for the heatmap colors
            color=color_source, 
            
            # Use the Original column for the tooltip text
            hover_data=['ID', color_col],
            
            color_continuous_scale='RdBu',
            template='plotly_dark', 
            title=title
        )
        
        # Clean up the Color Bar Label (remove "_scaled")
        fig.update_layout(coloraxis_colorbar=dict(title=color_col))
        
        fig.update_traces(marker=dict(size=5, opacity=0.8))
    else:
        fig = px.scatter(df, x='Latent_X', y='Latent_Y', hover_name='ID', template='plotly_dark', title=title)
        fig.update_traces(marker=dict(color='steelblue', size=6))

    if highlight_id and highlight_id in df['ID'].values:
        target = df[df['ID'] == highlight_id]
        fig.add_trace(go.Scatter(x=target['Latent_X'], y=target['Latent_Y'], mode='markers',
            marker=dict(size=12, color='red', line=dict(color='white', width=2)), name='Selected'))

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_traces(customdata=df['ID'])
    return fig

@app.callback(
    [Output('track-graph', 'figure'),
     Output('trace-histogram', 'figure'),
     Output('fitness-table-container', 'children')],
    [Input('latent-graph', 'clickData'),
     Input('track-metric-dropdown', 'value')]
)
def update_details(clickData, track_metric):
    empty_track = go.Figure().update_layout(template='plotly_dark', xaxis={'visible': False}, yaxis={'visible': False}, title="Select a point")
    empty_hist = go.Figure().update_layout(template='plotly_dark', xaxis={'visible': False}, yaxis={'visible': False}, title="No Data")
    empty_table = html.P("No selection", style={'color': '#888'})
    
    if not clickData:
        return empty_track, empty_hist, empty_table

    try:
        sel_id = str(clickData['points'][0]['customdata'])
        if sel_id not in tracks_dict:
            return empty_track, empty_hist, html.P(f"ID {sel_id} not found")

        track_arr = np.array(tracks_dict[sel_id])
        fit_entry = fitness_data[sel_id].item() if (fitness_data and sel_id in fitness_data) else {}
        trace_data = fit_entry.get(track_metric, []) if (track_metric and track_metric != "None") else []
        
        # Determine fixed range for the selected metric
        m_range = TRACE_RANGES.get(track_metric, [None, None])

        # --- A. TRACK PLOT ---
        track_fig = go.Figure()
        if len(trace_data) > 0:
            color_vals = interpolate_metrics_to_track(track_arr, trace_data)
            if color_vals is not None:
                track_fig.add_trace(go.Scatter(
                    x=track_arr[:, 0], y=track_arr[:, 1], mode='markers+lines',
                    line=dict(width=1, color='rgba(150,150,150,0.3)'),
                    marker=dict(
                        size=5, color=color_vals, colorscale=custom_burd,
                        showscale=True, cmin=m_range[0], cmax=m_range[1]
                    ),
                    name='Track'
                ))
            else:
                 track_fig.add_trace(go.Scatter(x=track_arr[:,0], y=track_arr[:,1], mode='lines', line=dict(color='white')))
        else:
            track_fig.add_trace(go.Scatter(x=track_arr[:,0], y=track_arr[:,1], mode='lines', line=dict(color='white')))

        track_fig.add_trace(go.Scatter(x=[track_arr[0,0]], y=[track_arr[0,1]], mode='markers', marker=dict(color='green', size=8), name='Start'))
        track_fig.update_layout(title=f"Track {sel_id}", template="plotly_dark", yaxis=dict(scaleanchor="x", scaleratio=1), margin=dict(l=10, r=10, t=30, b=10))

        # --- B. HISTOGRAM ---
        hist_fig = go.Figure()
        if len(trace_data) > 0:
            raw_vals = np.array(trace_data)[:, 0]
            # Set fixed bins based on range to prevent flickering bar widths
            bins_config = None
            if m_range[0] is not None:
                bins_config = dict(start=m_range[0], end=m_range[1], size=(m_range[1]-m_range[0])/40)

            hist_fig.add_trace(go.Histogram(x=raw_vals, marker_color='#636efa', opacity=0.75, xbins=bins_config))
            hist_fig.update_layout(
                title=f"Dist: {track_metric}", template="plotly_dark", 
                margin=dict(l=30, r=10, t=30, b=30), bargap=0.1,
                xaxis=dict(range=m_range) # Fixed X-Axis Scale
            )
        else:
            hist_fig.update_layout(title="No trace data", template="plotly_dark", xaxis={'visible': False}, yaxis={'visible': False})

        # --- C. SCALAR TABLE ---
        table_rows = []
        if fit_entry:
            for k in sorted(fit_entry.keys()):
                val = fit_entry[k]
                display_val = f"{val:.4f}" if isinstance(val, (int, float)) else (f"Trace ({len(val)})" if isinstance(val, (list, np.ndarray)) else str(val))
                color = '#4fd6ff' if isinstance(val, (int, float)) else '#ddd'
                table_rows.append(html.Tr([
                    html.Td(k, style={'padding': '2px 10px', 'fontWeight': 'bold', 'borderBottom': '1px solid #333'}),
                    html.Td(display_val, style={'padding': '2px 10px', 'textAlign': 'right', 'borderBottom': '1px solid #333', 'color': color})
                ]))
            table_comp = html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'fontFamily': 'monospace'})
        else:
            table_comp = html.P("No fitness data found for this ID.")

        return track_fig, hist_fig, table_comp

    except Exception as e:
        print(f"Callback Error: {e}")
        return empty_track, empty_hist, html.P("Error processing data")

if __name__ == '__main__':
    app.run(debug=True, port=8066)