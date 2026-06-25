"""Default parameters and configurations for the Metrics VAE.

All values are sourced from the module defaults.  ``None`` marks parameters
that must be supplied at runtime (e.g. device, dataset-dependent values).
"""

# ── Model Architecture ────────────────────────────────────────────────────────

MODEL_CONFIG = {
    # Input feature dimension: [speed, steering, distanceToBorder]
    "input_dim": 3,

    # Number of hidden channels in the CNN encoder / decoder
    "hidden_dim": 128,

    # Dimensionality of the latent space z
    "latent_dim": 32,

    # Number of stacked CircularResBlocks in encoder and decoder
    "n_layers": 5,

    # Convolutional kernel size used in every CircularConv1d
    "kernel_size": 7,

    # Maximum sequence length supported by the positional encoding
    "max_seq_len": 5000,

    # Number of DFT frequency bins used for the power-pooling step
    "freq_bins": 64,
}

# Weight-initialisation overrides applied after model construction
MODEL_INIT = {
    # fc_var bias is clamped low at start to avoid KLD explosion in epoch 1
    "fc_var_bias_init": -2.0,
    # fc_var weight uses orthogonal init with a small gain
    "fc_var_weight_gain": 0.01,
}

# ── Training Hyper-parameters ─────────────────────────────────────────────────

TRAINING_CONFIG = {
    # Adam optimiser learning rate
    "lr": 5e-4,

    # Maximum number of training epochs
    "epochs": 700,

    # Early-stopping: epochs with no val-loss improvement before stopping
    "patience": 50,

    # Early-stopping: minimum improvement to reset the patience counter
    "min_delta": 0.01,

    # Gradient clipping max-norm
    "max_grad_norm": 0.5,

    # ── Cyclical KLD annealing ──────────────────────────────────────────────
    # Number of annealing cycles over the full training run
    "kld": {
        "n_cycles": 1,

        # Peak beta value (weight on the KLD term)
        "max_beta": 0.02,

        # Fraction of each cycle used for the linear warm-up ramp
        "ratio": 0.3,
    },

    # ── Learning-rate scheduler (ReduceLROnPlateau) ─────────────────────────
    "lr_schedule": {
        # Multiplicative factor applied to LR on plateau
        "factor": 0.5,

        # Epochs with no improvement before reducing LR
        "patience": 25,

        # Floor value for the learning rate
        "min_lr": 1e-5,
    },

    # Per-feature loss weights [speed, steering, distanceToBorder].
    # None → all features weighted equally.
    "dim_weights": None,
}

# ── Fine-tuning Hyper-parameters ─────────────────────────────────────────────

FINETUNING_CONFIG = {
    # Adam learning rate for the unfrozen encoder layers (deep blocks + fc_mu/fc_var).
    "lr": 5e-4,

    # Adam learning rate for the decoder.
    # Set to a value lower than `lr` to let the decoder track the encoder's
    # shifting z-distribution without dominating the reconstruction signal.
    # Set to 0.0 to use the same lr as the encoder (no differential LR).
    "decoder_lr": 0.0,

    # Maximum number of fine-tuning epochs (early stopping usually kicks in sooner)
    "epochs": 50,

    # Early-stopping patience during fine-tuning
    "patience": 20,

    # Batch size for the fine-tuning DataLoaders. Fine-tuning runs on the small
    # elite set, so a smaller batch than pretraining (DATA_CONFIG["batch_size"])
    # gives more gradient updates per epoch. Capped at the number of elites.
    "batch_size": 32,
    
    "dim_weights": [0.25417647, 2.6773722, 0.06845125],

    # Number of encoder CircularResBlocks to freeze (counting from block 0,
    # i.e. the lowest-dilation blocks that learn generic local patterns).
    "n_frozen_encoder_blocks": 2,

    # ── Cyclical KLD annealing ──────────────────────────────────────────────
    "kld": {
        # Single cycle — beta restarts from 0 to avoid latent-space disruption
        "n_cycles": 1,

        # Reduced peak beta: recon loss dominates to preserve latent structure
        "max_beta": 0.05,

        # Longer warm-up fraction so beta rises slowly
        "ratio": 0.5,
    },
}

# ── Loss Function ─────────────────────────────────────────────────────────────

LOSS_CONFIG = {
    # Default beta (KLD weight); overridden each epoch by cyclical annealing
    "beta": 0.0,

    # log_var clamping range to prevent numerical instability
    "log_var_clamp_min": -20.0,
    "log_var_clamp_max": 10.0,

    # Per-feature loss weights (None → uniform)
    "dim_weights": None,
}

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_CONFIG = {
    # Value used to pad variable-length sequences in a batch
    "padding_value": 0.0,

    # Batch size is not fixed here; set it where the DataLoader is created
    "batch_size": 128,

    # Train / validation split ratio (not enforced in code; reference only)
    "val_split": 0.2,
}

# ── Preprocessing (MetricsPreprocessor) ──────────────────────────────────────

PREPROCESSING_CONFIG = {
    # Raw input: 7 columns  →  [id, Speed, Steering, Accel, Brake, Gear, distanceToBorder]
    "raw_input_cols": 7,

    # Columns dropped before feeding the model  (id=0, accel=3, brake=4, gear=5)
    "drop_cols": [0, 3, 4, 5],

    # Retained output columns (indices in the *original* array)
    "keep_cols": {
        "speed": 1,
        "steering": 2,
        "distanceToBorder": 6,
    },

    # Column indices in the *processed* 3-column array
    "processed_col_indices": {
        "speed": 0,
        "steering": 1,
        "position": 2,
    },

    # Speed normalisation: divide by MAX_SPEED, clip to [0, 1]
    "max_speed": 65,

    # Steering normalisation: multiply by 0.5  →  range [-0.5, 0.5]
    "steering_scale": 0.5,

    # Track width used to centre position  →  output range [-0.5, 0.5]
    "track_width": 18,

    # Minimum per-feature std required to consider a sample valid
    "min_std": 1e-6,
}
