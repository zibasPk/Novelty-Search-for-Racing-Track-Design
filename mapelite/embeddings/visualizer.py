import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASETS_FOLDER = Path(__file__).parent / "datasets/embeddings/good"
TRACKS_FILE = Path(__file__).parent / "datasets/tracks.npz"
DEFAULT_MAX_POINTS = 9000

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_available_datasets():
    """Scan the datasets folder and return a list of available .npz files."""
    if not DATASETS_FOLDER.exists():
        print(f"Warning: Datasets folder '{DATASETS_FOLDER}' not found.")
        return []
    return sorted([f.name for f in DATASETS_FOLDER.glob("*.npz")])

def load_dataset(filename):
    """Load a dataset file and return embeddings and IDs."""
    filepath = DATASETS_FOLDER / filename
    if not filepath.exists():
        return None, None
    data = np.load(filepath)
    embeddings = data["embeddings"]
    ids = data["ids"]
    return embeddings, ids

def prepare_dataframe(latents, raw_ids, max_points=DEFAULT_MAX_POINTS):
    """Prepare and optionally downsample the dataframe."""
    df = pd.DataFrame(latents, columns=['Latent_X', 'Latent_Y'])
    df['ID'] = [str(x) for x in raw_ids]
    
    if max_points and len(df) > max_points:
        print(f"Downsampling from {len(df)} to {max_points} to prevent WebGL crash...")
        return df.sample(n=max_points, random_state=42)
    return df

def create_latent_figure(df_plot, title_suffix="", highlight_id=None):
    """Create the latent space scatter plot figure."""
    # Create color column for highlighting
    if highlight_id:
        df_plot = df_plot.copy()
        df_plot['highlight'] = df_plot['ID'].apply(lambda x: 'Searched' if x == highlight_id else 'Normal')
        color_map = {'Searched': 'red', 'Normal': 'steelblue'}
        fig = px.scatter(
            df_plot, 
            x='Latent_X', 
            y='Latent_Y', 
            hover_name='ID',
            color='highlight',
            color_discrete_map=color_map,
            title=f"Latent Space ({len(df_plot)} samples){title_suffix}",
            template="plotly_dark",
            render_mode='webgl'
        )
        # Make searched point larger
        fig.for_each_trace(
            lambda trace: trace.update(marker=dict(size=15, opacity=1.0)) if trace.name == 'Searched' else trace.update(marker=dict(size=8, opacity=0.7))
        )
    else:
        fig = px.scatter(
            df_plot, 
            x='Latent_X', 
            y='Latent_Y', 
            hover_name='ID',
            title=f"Latent Space ({len(df_plot)} samples){title_suffix}",
            template="plotly_dark",
            render_mode='webgl'
        )
        fig.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='White'))
        )
    
    fig.update_traces(customdata=df_plot['ID'])
    fig.update_layout(
        clickmode='event+select',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ==========================================
# 3. LOAD INITIAL DATA
# ==========================================
print("Loading data...")

# Load tracks dictionary
raw_tracks_file = np.load(TRACKS_FILE, allow_pickle=True)
tracks_dict = dict(raw_tracks_file)
print(f"Loaded {len(tracks_dict)} tracks.")

# Get available datasets
available_datasets = get_available_datasets()
print(f"Found {len(available_datasets)} datasets: {available_datasets}")

# Load initial dataset (first one available, or empty)
initial_df = pd.DataFrame(columns=['Latent_X', 'Latent_Y', 'ID'])
initial_dataset = available_datasets[0] if available_datasets else None

if initial_dataset:
    latents, raw_ids = load_dataset(initial_dataset)
    if latents is not None:
        initial_df = prepare_dataframe(latents, raw_ids)
        print(f"Initial dataset '{initial_dataset}': {len(initial_df)} samples")

# ==========================================
# 4. INITIALIZE DASH APP
# ==========================================
app = dash.Dash(__name__)

# Create dropdown options
dropdown_options = [{'label': ds, 'value': ds} for ds in available_datasets]

# ==========================================
# 5. LAYOUT
# ==========================================
app.layout = html.Div([
    html.H1("Latent Space Explorer", style={'fontFamily': 'Arial', 'textAlign': 'center', 'color': '#333'}),
    
    # Controls Row
    html.Div([
        # Dataset Selector
        html.Div([
            html.Label("Dataset: ", style={'fontWeight': 'bold', 'marginRight': '10px', 'color': '#333'}),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=dropdown_options,
                value=initial_dataset,
                style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        # Sample Count Selector
        html.Div([
            html.Label("Max Samples: ", style={'fontWeight': 'bold', 'marginRight': '10px', 'color': '#333'}),
            dcc.Input(
                id='sample-count-input',
                type='number',
                value=DEFAULT_MAX_POINTS,
                min=100,
                max=50000,
                step=100,
                style={'width': '100px', 'verticalAlign': 'middle'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),
        
        # ID Search
        html.Div([
            html.Label("Search ID: ", style={'fontWeight': 'bold', 'marginRight': '10px', 'color': '#333'}),
            dcc.Input(
                id='id-search-input',
                type='text',
                placeholder='Enter track ID...',
                debounce=True,
                style={'width': '200px', 'verticalAlign': 'middle'}
            ),
            html.Span(id='search-status', style={'marginLeft': '10px', 'color': '#666'})
        ], style={'display': 'inline-block'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        # Left: Latent Space
        html.Div([
            dcc.Graph(id='latent-graph', figure=create_latent_figure(initial_df), style={'height': '70vh'})
        ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right: Track View
        html.Div([
            dcc.Graph(id='track-graph', style={'height': '70vh'})
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
    ], style={'display': 'flex', 'flexDirection': 'row'})
])

# ==========================================
# 6. CALLBACKS
# ==========================================

# Callback to update latent graph when dataset, sample count, or search changes
@app.callback(
    [Output('latent-graph', 'figure'),
     Output('search-status', 'children')],
    [Input('dataset-dropdown', 'value'),
     Input('sample-count-input', 'value'),
     Input('id-search-input', 'value')]
)
def update_latent_graph(selected_dataset, max_samples, search_id):
    """Load the selected dataset and update the latent space plot."""
    search_status = ""
    
    if not selected_dataset:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No dataset selected",
            template="plotly_dark",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return empty_fig, search_status
    
    latents, raw_ids = load_dataset(selected_dataset)
    if latents is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"Error loading dataset: {selected_dataset}",
            template="plotly_dark",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return empty_fig, search_status
    
    # Use provided max_samples or default
    max_points = max_samples if max_samples else DEFAULT_MAX_POINTS
    df_plot = prepare_dataframe(latents, raw_ids, max_points=max_points)
    print(f"Loaded dataset '{selected_dataset}': {len(df_plot)} samples")
    
    # Handle ID search
    highlight_id = None
    if search_id:
        search_id = str(search_id).strip()
        if search_id in df_plot['ID'].values:
            highlight_id = search_id
            search_status = f"✓ Found ID: {search_id}"
        else:
            # Check if it exists in the full dataset but was downsampled out
            all_ids = [str(x) for x in raw_ids]
            if search_id in all_ids:
                search_status = f"⚠ ID exists but not in current sample. Try increasing max samples."
            else:
                search_status = f"✗ ID not found: {search_id}"
    
    return create_latent_figure(df_plot, f" - {selected_dataset}", highlight_id=highlight_id), search_status

# Callback to display track on click
@app.callback(
    Output('track-graph', 'figure'),
    Input('latent-graph', 'clickData')
)
def display_track(clickData):
    # Empty state configuration
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Click a point on the left to view track", 
        template="plotly_dark",
        xaxis={'visible': False}, 
        yaxis={'visible': False}
    )

    if not clickData:
        return empty_fig

    try:
        # 1. Retrieve ID from graph click
        selected_id = clickData['points'][0]['customdata']
        lookup_key = str(selected_id)

        # 2. Lookup in Dictionary
        if lookup_key not in tracks_dict:
            empty_fig.update_layout(title=f"Error: ID '{lookup_key}' not found in tracks.npz")
            return empty_fig
        
        # 3. Plot Track
        track_data = tracks_dict[lookup_key]
        track_arr = np.array(track_data)
        
        # Check if track has data
        if track_arr.size == 0:
            empty_fig.update_layout(title=f"Track ID {lookup_key} is empty")
            return empty_fig

        fig_track = px.line(
            x=track_arr[:, 0], 
            y=track_arr[:, 1], 
            title=f"Track ID: {lookup_key}",
            template="plotly_dark",
            markers=False # False makes it cleaner, set True to see segments
        )
        
        # Mark Start and End
        fig_track.add_trace(go.Scatter(x=[track_arr[0,0]], y=[track_arr[0,1]], mode='markers', marker=dict(color='green', size=10), name='Start'))
        fig_track.add_trace(go.Scatter(x=[track_arr[-1,0]], y=[track_arr[-1,1]], mode='markers', marker=dict(color='red', size=10), name='End'))
        
        # Ensure aspect ratio is correct (so round tracks look round)
        fig_track.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig_track
        
    except Exception as e:
        print(f"Error in callback: {e}")
        empty_fig.update_layout(title=f"Error displaying track")
        return empty_fig

# ==========================================
# 7. RUN
# ==========================================
if __name__ == '__main__':
    # Using specific port and turning off hot-reload can sometimes help with context issues
    app.run(debug=True, port=8066, use_reloader=True)