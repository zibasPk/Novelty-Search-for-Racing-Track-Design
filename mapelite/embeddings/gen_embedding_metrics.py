import os
import torch
import torch.nn as nn
import numpy as np
import math
import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. Model Definitions (Must match training)
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

class MetricsTransformerVAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=2, n_layers=4, n_heads=4, max_seq_len=200):
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
        log_var = self.fc_var(summary)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len, src_key_padding_mask=None):
        batch_size = z.shape[0]
        z_projected = self.decoder_input(z).unsqueeze(1)
        queries = self.pos_encoding.pe[:, 1 : seq_len + 1, :]
        queries = queries.expand(batch_size, -1, -1)
        x = queries + z_projected
        x = self.transformer_decoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.final_projection(x)
        return output
    
    def forward(self, x, src_key_padding_mask=None):
        mu, log_var = self.encode(x, src_key_padding_mask)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.shape[1], src_key_padding_mask)
        return recon_x, mu, log_var

# ==========================================
# 2. Data Utilities
# ==========================================

class MetricsDataset(Dataset):
    def __init__(self, metrics):
        self.metrics = [torch.tensor(t, dtype=torch.float32) for t in metrics]

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

def collate_fn(batch):
    padded_metrics = pad_sequence(batch, batch_first=True, padding_value=0.0)
    mask = (padded_metrics.abs().sum(dim=-1) == 0)    
    return padded_metrics, mask

def load_and_preprocess_data(path):
    print(f"Loading data from {path}...")
    dataset = np.load(path)
    flat_metrics = dataset["data"]
    indices = dataset["indices"]
    ids = dataset["ids"]

    # --- Constants matching Notebook ---
    SPEED_IDX = 0
    STEERING_IDX = 1
    POS_IDX = 2 
    MAX_SPEED = 65
    TRACK_WIDTH = 18
    STEERING_THRESHOLD = 0.05

    # 1. Column Selection: ['Speed', 'Steering', 'Track_Angle']
    # Removing [0 (id), 3 (accel), 4 (brake), 5 (gear)]
    flat_metrics = np.delete(flat_metrics, [0,3,4,5], axis=1)

    # 2. Normalization
    # Speed
    flat_metrics[:, SPEED_IDX] = flat_metrics[:, SPEED_IDX] / MAX_SPEED
    
    # Position (Clip and Center)
    flat_metrics[:, POS_IDX] = np.clip(flat_metrics[:, POS_IDX], 0, TRACK_WIDTH)
    normalized_pos = flat_metrics[:, POS_IDX] / TRACK_WIDTH
    # Transform to: -0.5 (Left) -> 0.0 (Center) -> +0.5 (Right)
    flat_metrics[:, POS_IDX] = normalized_pos - 0.5

    print(f"Processed metrics shape (pre-split): {flat_metrics.shape}")

    # 3. Split into individual tracks
    metrics = np.split(flat_metrics, indices)

    # 4. CANONICALIZATION (Matching the Notebook Logic)
    # Ensure all tracks start with a Right Turn to stop mirroring issues
    for track in metrics:
        if len(track) == 0: continue

        significant_mask = np.abs(track[:, STEERING_IDX]) > STEERING_THRESHOLD
        significant_indices = np.where(significant_mask)[0]
        
        if len(significant_indices) > 0:
            first_idx = significant_indices[0]
            first_val = track[first_idx, STEERING_IDX]
            
            # If track starts with Left Turn, MIRROR the whole world
            if first_val < 0:
                track[:, STEERING_IDX] *= -1  # Flip Steering
                track[:, POS_IDX] *= -1       # Flip Position (Left side becomes Right side)

    # 5. Clean up (Remove NaNs/Inf)
    keep = [np.isfinite(m).all() for m in metrics]
    metrics_clean = [m for i, m in enumerate(metrics) if keep[i]]
    ids_clean = ids[keep]
    
    print(f"Kept {len(ids_clean)} samples after cleaning.")
    
    return metrics_clean, ids_clean

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
        
        # Default UMAP parameters (can be overridden via kwargs)
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
        
        # Print explained variance
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
# 4. Main Generator Logic
# ==========================================

def load_model(path, device):
    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    model = MetricsTransformerVAE(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        max_seq_len=config['max_seq_len']
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded with latent_dim={config['latent_dim']}")
    return model, config['latent_dim']

def main():
    parser = argparse.ArgumentParser(description="Generate Track Embeddings from VAE with Dimensionality Reduction")
    parser.add_argument("--data", type=str, default="datasets/dataset10k_metrics.npz", 
                        help="Path to input .npz dataset")
    parser.add_argument("--model", type=str, default="models/model_metrics_VAE/model_metrics_VAE_latent32.pth", 
                        help="Path to trained .pth model")
    parser.add_argument("--output", type=str, default="track_embeddings_metrics.npz", 
                        help="Path to save output .npz")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Inference batch size")
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

    # 1. Load Data
    metrics_list, ids_list = load_and_preprocess_data(args.data)
    
    # 2. Load Model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found at {args.model}")
        
    model, latent_dim = load_model(args.model, device)
    
    # 3. Prepare Loader
    dataset = MetricsDataset(metrics_list)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 4. Generate Embeddings
    embeddings_list = []
    print("Generating embeddings...")
    
    with torch.no_grad():
        for data, mask in tqdm(loader, desc="Encoding"):
            data = data.to(device)
            mask = mask.to(device)
            mu, _ = model.encode(data, src_key_padding_mask=mask)
            embeddings_list.append(mu.cpu().numpy())
    
    original_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Original embeddings shape: {original_embeddings.shape}")
    
    # 5. Dimensionality Reduction (if needed)
    save_dict = {'ids': ids_list}
    
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
    
    # 6. Save
    print(f"\nSaving to {args.output}...")
    for key, val in save_dict.items():
        if key != 'ids':
            print(f"  {key}: {val.shape}")
    
    np.savez_compressed(args.output, **save_dict)
    print("Done.")

if __name__ == "__main__":
    main()