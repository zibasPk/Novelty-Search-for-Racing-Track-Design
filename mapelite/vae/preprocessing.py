"""Data preprocessing for metrics-based track embeddings."""

import numpy as np


class MetricsPreprocessor:
    """Preprocesses raw simulation telemetry into model-ready features.

    Expected input: 2D array with columns
    [id, Speed, Steering, Accel, Brake, Gear, distanceToBorder].

    Output: 2D array with columns [Speed, Steering, distanceToBorder],
    normalised to match the VAE training distribution.
    Note: Canonicalisation (shift alignment/mirroring) is no longer needed
    as the Circular VAE handles shift invariance structurally.
    """

    SPEED_IDX = 0
    STEERING_IDX = 1
    POS_IDX = 2

    MAX_SPEED = 65
    TRACK_WIDTH = 18

    # Columns to drop: id (0), accel (3), brake (4), gear (5)
    # columns to keep: speed (1), steering (2), distanceToBorder (6)
    DROP_COLS = [0, 3, 4, 5]

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess *input_data* in-place-safe fashion (copies first)."""
        data = input_data.copy()
        
        # 1. Validate raw data shape/finiteness
        self._validate_raw(data)
        
        # 2. Select the 3 active columns
        data = self._select_columns(data)
        
        # 3. Validate the processed signal (e.g. check for dead signals)
        self._validate_signal(data)
        
        # 4. Normalise
        data = self._normalise(data)
        
        return data

    @staticmethod
    def is_valid(processed_metric: np.ndarray) -> bool:
        """
        Equivalent to the notebook's `is_valid_sample`.
        Checks if a processed (3-column) metric array is healthy.
        Useful for boolean masking/filtering a dataset before training.
        """
        if not np.isfinite(processed_metric).all():
            return False
        if np.any(processed_metric.std(axis=0) < 1e-6):
            return False
        return True

    # -- private helpers ------------------------------------------------------

    @staticmethod
    def _validate_raw(data: np.ndarray) -> None:
        if data.ndim != 2 or data.shape[1] != 7:
            raise ValueError(
                f"Expected 2D array with 7 columns "
                f"[id, Speed, Steering, Accel, Brake, Gear, distanceToBorder], "
                f"but got shape {data.shape}"
            )
        if len(data) == 0:
            raise ValueError("Input data is empty.")
        if not np.isfinite(data).all():
            raise ValueError(
                "Input data contains NaN or Inf values. "
                "Please clean the data before preprocessing."
            )

    def _validate_signal(self, data: np.ndarray) -> None:
        """Ensures the selected features actually contain movement/variance."""
        if np.any(data.std(axis=0) < 1e-6):
            raise ValueError(
                "Input data contains static features (std < 1e-6). "
                "The car appears to be stuck or the recording is invalid."
            )

    def _select_columns(self, data: np.ndarray) -> np.ndarray:
        return np.delete(data, self.DROP_COLS, axis=1)

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        # Speed → Divide by max speed and clip to [0.0, 1.0]
        data[:, self.SPEED_IDX] /= self.MAX_SPEED
        data[:, self.SPEED_IDX] = np.clip(data[:, self.SPEED_IDX], 0.0, 1.0)

        # Steering → scale to be between -0.5 and 0.5
        data[:, self.STEERING_IDX] *= 0.5

        # Position → clip to track width, centre around 0
        # Transform to: -0.5 (Left Edge) -> 0.0 (Center) -> +0.5 (Right Edge)
        data[:, self.POS_IDX] = np.clip(
            data[:, self.POS_IDX], 0, self.TRACK_WIDTH
        )
        data[:, self.POS_IDX] = data[:, self.POS_IDX] / self.TRACK_WIDTH - 0.5
        
        return data