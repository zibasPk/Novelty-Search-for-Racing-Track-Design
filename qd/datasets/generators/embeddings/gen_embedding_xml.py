import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import os
import argparse
from tqdm import tqdm

# ==========================================
# 1. Model Definition (Must match training)
# ==========================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TrackTransformerVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=2, n_layers=4, n_heads=4, max_seq_len=200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder parts (needed for loading state_dict, even if not used for embedding)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.final_projection = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, src_key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x) 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        summary = x[:, 0, :] 
        mu = self.fc_mu(summary)
        return mu

# ==========================================
# 2. Data Loading & Preprocessing Helpers
# ==========================================

class TrackDataset(Dataset):
    def __init__(self, tracks, ids):
        self.tracks = [torch.tensor(t, dtype=torch.float32) for t in tracks]
        self.ids = ids

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx], self.ids[idx]

def collate_fn(batch):
    tracks = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    
    padded_tracks = pad_sequence(tracks, batch_first=True, padding_value=0.0)
    mask = (padded_tracks.abs().sum(dim=-1) == 0)
    
    return padded_tracks, mask, np.array(ids)

def is_valid_sample(track):
    return np.isfinite(track).all()

def canonicalize_tracks(tracks):
    canonical_tracks = []
    for t in tracks:
        if np.sum(t[:, 1]) < 0:
            t = t.copy()
            t[:, 1] = -t[:, 1]
        canonical_tracks.append(t)
    return canonical_tracks

# ==========================================
# 3. Dimensionality Reduction
# ==========================================

def reduce_dimensions(embeddings, method='umap', target_dim=2, random_state=42, **kwargs):
    """
    Reduce embeddings to target_dim dimensions.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: 'umap', 'pca', or 'tsne'
        target_dim: target number of dimensions (default: 2)
        random_state: random seed for reproducibility
        **kwargs: additional arguments for the reduction method
    
    Returns:
        reduced_embeddings: numpy array of shape (n_samples, target_dim)
    """
    n_samples, n_features = embeddings.shape
    
    print(f"\nReducing from {n_features}D to {target_dim}D using {method.upper()}...")
    
    if n_features <= target_dim:
        print(f"Warning: latent_dim ({n_features}) <= target_dim ({target_dim}). No reduction needed.")
        return embeddings
    
    if method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'random_state': random_state
        }
        umap_params.update(kwargs)
        
        reducer = umap.UMAP(n_components=target_dim, **umap_params)
        reduced = reducer.fit_transform(embeddings)
        
    elif method == 'pca':
        from sklearn.decomposition import PCA
        
        pca_params = {'random_state': random_state}
        pca_params.update(kwargs)
        
        reducer = PCA(n_components=target_dim, **pca_params)
        reduced = reducer.fit_transform(embeddings)
        
        explained_var = np.sum(reducer.explained_variance_ratio_) * 100
        print(f"PCA explained variance: {explained_var:.2f}%")
        
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        
        tsne_params = {
            'perplexity': 30,
            'learning_rate': 200,
            'random_state': random_state
        }
        tsne_params.update(kwargs)
        
        reducer = TSNE(n_components=target_dim, **tsne_params)
        reduced = reducer.fit_transform(embeddings)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'umap', 'pca', or 'tsne'")
    
    print(f"Reduction complete. Output shape: {reduced.shape}")
    return reduced

# ==========================================
# 4. Main Execution
# ==========================================

def load_model(model_path, device):
    """Load model and return model instance with latent dimension."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract dimensions from checkpoint
    latent_dim = checkpoint.get('latent_dim', 2)
    hidden_dim = checkpoint.get('hidden_dim', 64)
    n_layers = checkpoint.get('n_layers', 4)
    n_heads = checkpoint.get('n_heads', 4)
    
    model = TrackTransformerVAE(
        input_dim=2,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        n_heads=n_heads
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    return model, latent_dim

def load_and_preprocess_data(data_path):
    """Load and preprocess track data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print("Loading and preprocessing data...")
    dataset_npz = np.load(data_path)
    flat_tracks = dataset_npz["data"]
    indices = dataset_npz["indices"]
    raw_ids = dataset_npz["ids"]

    # Split flat array into list of tracks
    tracks = np.split(flat_tracks, indices)

    # Filter invalid tracks
    keep = [is_valid_sample(track) for track in tracks]
    tracks_clean = [track for i, track in enumerate(tracks) if keep[i]]
    ids_clean = raw_ids[keep]
    
    print(f"Original: {len(tracks)}, Cleaned: {len(tracks_clean)}")

    # Canonicalize (Mirroring invariance)
    tracks_clean = canonicalize_tracks(tracks_clean)

    # Log Normalization (CRITICAL: Must match training)
    for track in tracks_clean:
        track[:, 0] = np.log1p(track[:, 0])

    return tracks_clean, ids_clean

def main():
    parser = argparse.ArgumentParser(description="Generate Track Embeddings from VAE with Dimensionality Reduction")
    parser.add_argument("--data", type=str, default="datasets/dataset10k_xml.npz",
                        help="Path to input .npz dataset")
    parser.add_argument("--model", type=str, default="models/model_metrics_VAE/model_xml_VAE_latent32.pth",
                        help="Path to trained .pth model")
    parser.add_argument("--output", type=str, default="datasets/embeddings/track_embeddings_xml_32dim_mu.npz",
                        help="Path to save output .npz")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--reduce_method", type=str, default='umap', choices=['umap', 'pca', 'tsne'],
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--target_dim", type=int, default=2,
                        help="Target dimensionality after reduction (default: 2)")
    parser.add_argument("--no_reduce", action='store_true',
                        help="Skip dimensionality reduction and save original embeddings")
    parser.add_argument("--save_original", action='store_true',
                        help="Save both original and reduced embeddings")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model, latent_dim = load_model(args.model, device)

    # 2. Load and Preprocess Data
    tracks_clean, ids_clean = load_and_preprocess_data(args.data)

    # 3. Create DataLoader
    ds = TrackDataset(tracks_clean, ids_clean)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Generate Embeddings
    print("Generating embeddings...")
    
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for tracks_batch, mask_batch, ids_batch in tqdm(loader, desc="Encoding"):
            tracks_batch = tracks_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            mu = model.encode(tracks_batch, src_key_padding_mask=mask_batch)
            
            all_embeddings.append(mu.cpu().numpy())
            all_ids.append(ids_batch)

    original_embeddings = np.concatenate(all_embeddings, axis=0)
    final_ids = np.concatenate(all_ids, axis=0)
    
    print(f"Original embeddings shape: {original_embeddings.shape}")

    # 5. Dimensionality Reduction (if needed)
    save_dict = {'ids': final_ids}
    
    if args.no_reduce:
        print("Skipping dimensionality reduction as requested.")
        save_dict['embeddings'] = original_embeddings
    else:
        if latent_dim > args.target_dim:
            reduced_embeddings = reduce_dimensions(
                original_embeddings,
                method=args.reduce_method,
                target_dim=args.target_dim,
                random_state=args.random_seed
            )
            
            if args.save_original:
                save_dict['embeddings_original'] = original_embeddings
                save_dict['embeddings_reduced'] = reduced_embeddings
                save_dict['embeddings'] = reduced_embeddings  # Default to reduced
            else:
                save_dict['embeddings'] = reduced_embeddings
        else:
            print(f"Latent dim ({latent_dim}) <= target dim ({args.target_dim}). Using original embeddings.")
            save_dict['embeddings'] = original_embeddings
            
        if args.save_original and latent_dim > args.target_dim:
            save_dict['embeddings_original'] = original_embeddings

    # 6. Save Results
    print(f"\nSaving {len(final_ids)} embeddings to {args.output}...")
    for key, val in save_dict.items():
        if key != 'ids':
            print(f"  {key}: {val.shape}")
    
    np.savez_compressed(args.output, **save_dict)
    print("Done.")

if __name__ == "__main__":
    main()