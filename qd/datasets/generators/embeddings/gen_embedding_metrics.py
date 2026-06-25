import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from qd.vae import MetricsDataset, MetricsPreprocessor, MetricsVAE
from qd.vae.data import collate_fn

# ==========================================
# 1. Data Utilities
# ==========================================

def load_and_preprocess_data(path):
    print(f"Loading data from {path}...")
    dataset = np.load(path)
    flat_metrics = dataset["data"]
    indices = dataset["indices"]
    ids = dataset["ids"]

    preprocessor = MetricsPreprocessor()

    # Split into individual tracks and preprocess each using shared VAE utility.
    raw_tracks = np.split(flat_metrics, indices)
    metrics_clean = []
    ids_clean = []

    for track, track_id in zip(raw_tracks, ids):
        if len(track) == 0:
            continue

        try:
            metrics_clean.append(preprocessor(track))
            ids_clean.append(track_id)
        except ValueError:
            # Skip invalid tracks (empty, non-finite, malformed shape).
            continue

    ids_clean = np.asarray(ids_clean)
    
    print(f"Kept {len(ids_clean)} samples after cleaning.")
    
    return metrics_clean, ids_clean

# ==========================================
# 2. Main Generator Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate original track embeddings from VAE")
    parser.add_argument("--data", type=str, default="qd/datasets/training/dataset20k_metrics_mixedRng_tita_canonicized.npz", 
                        help="Path to input .npz dataset")
    parser.add_argument("--model", type=str, default="qd/pretrained_models/model_metrics_VAE/model_metrics_VAE_mixRng_tita_circular_canon_1.pth", 
                        help="Path to trained .pth model")
    parser.add_argument("--output", type=str, default="qd/datasets/embeddings/track_embeddings_metrics_32dim_rngMixDS_tita_circular_canon_1.npz", 
                        help="Path to save output .npz")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Inference batch size")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    metrics_list, ids_list = load_and_preprocess_data(args.data)
    
    # 2. Load Model
    model, _ , _ = MetricsVAE.load_pretrained(args.model, device)
    model.eval()
    
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
    print(f"Embeddings shape: {original_embeddings.shape}")

    save_dict = {
        'ids': ids_list,
        'embeddings': original_embeddings,
    }

    # 6. Save
    print(f"\nSaving to {args.output}...")
    for key, val in save_dict.items():
        if key != 'ids':
            print(f"  {key}: {val.shape}")
    
    np.savez_compressed(args.output, **save_dict)
    print("Done.")

if __name__ == "__main__":
    main()