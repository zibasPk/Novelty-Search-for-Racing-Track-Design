"""Data preprocessing for metrics-based track embeddings."""

import numpy as np


class MetricsPreprocessor:
    """Preprocesses raw simulation telemetry into model-ready features.

    Expected input: 2D array with columns
    [id, Speed, Steering, Accel, Brake, Gear, distanceToBorder].

    Output: 2D array with columns [Speed, Steering, distanceToBorder],
    normalised and canonicalised so that all tracks begin with a right turn.
    """

    SPEED_IDX = 0
    STEERING_IDX = 1
    POS_IDX = 2

    MAX_SPEED = 65
    TRACK_WIDTH = 18
    STEERING_THRESHOLD = 0.05

    # Columns to drop: time step (0), accel (3), brake (4), gear (5)
    # columns to keep: speed (1), steering (2), distanceToBorder (6)
    DROP_COLS = [0, 3, 4, 5]

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess *input_data* in-place-safe fashion (copies first)."""
        data = input_data.copy()
        self._validate(data)
        data = self._select_columns(data)
        data = self._normalise(data)
        # data = self._canonicalise(data)
        return data

    # -- private helpers ------------------------------------------------------

    @staticmethod
    def _validate(data: np.ndarray) -> None:
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

    def _select_columns(self, data: np.ndarray) -> np.ndarray:
        return np.delete(data, self.DROP_COLS, axis=1)

    def _normalise(self, data: np.ndarray) -> np.ndarray:
        # Speed → [0, 1]
        data[:, self.SPEED_IDX] /= self.MAX_SPEED

        # Position → clip to track width, centre around 0
        data[:, self.POS_IDX] = np.clip(
            data[:, self.POS_IDX], 0, self.TRACK_WIDTH
        )
        data[:, self.POS_IDX] = data[:, self.POS_IDX] / self.TRACK_WIDTH - 0.5
        return data

    def _canonicalise(self, data: np.ndarray) -> np.ndarray:
        """Mirror steering & position so the first significant turn is right."""
        significant = np.where(
            np.abs(data[:, self.STEERING_IDX]) > self.STEERING_THRESHOLD
        )[0]

        if len(significant) > 0 and data[significant[0], self.STEERING_IDX] < 0:
            data[:, self.STEERING_IDX] *= -1
            data[:, self.POS_IDX] *= -1

        return data
