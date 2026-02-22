import os

import numpy as np
import requests
import joblib
import torch


import embeddings.models.metrics_model as metrics_model
from utils import EMBEDDING_MODEL, pca_align
from abc import ABC, abstractmethod
from config import (
    BASE_URL, INVALID_SCORE
)


class Evaluator(ABC):
    """Abstract base class for solution evaluators."""
    @abstractmethod
    def evaluate(self, sol):
        """Evaluates a solution and returns (id, ok, msg, fitness_score, descriptor)."""
        pass

    def fitness_formula(self, fit):
        """Calculates the scalar fitness score based on evaluation metrics."""
        length = max(fit.get('length', 0.0), 1e-3)
        bend_len = fit.get('right_len', 0.0) + fit.get('left_len', 0.0)
        overtakes = fit.get('total_overtakes', 0.0)
        dx = abs(fit.get('deltaX', 0.0)) or 1e-3

        bend_ratio = bend_len / length

        score = overtakes
        return float(score)


class EvaluatorMAPElite(Evaluator):
    def __init__(self, model_path="data/EmbeddingModels/umap_model.joblib"):
        try:
            self.embedding_model = joblib.load(model_path)
        except FileNotFoundError:
            print(
                "Warning: UMAP model not found. Placeholder used. (Run the setup notebook first?)")
            # Placeholder class to prevent crash if model isn't trained/found

            class PlaceholderUMAP:
                def transform(self, data):
                    # Returns a dummy 2D descriptor
                    return np.zeros((data.shape[0], 2))
            self.embedding_model = PlaceholderUMAP()

    def descriptor_from_track(self, sol):
        """Converts the track's spline vector into the 2D behavioral descriptor using UMAP."""
        # The 'splineVector' is assumed to be part of the evaluation JSON response
        pts = np.array([[p["x"], p["y"]]
                       for p in sol.get("splineVector", [])], dtype=float)

        # Align the spline to account for rotation/translation invariance
        aligned = pca_align(pts)

        # Flatten the aligned points to create the feature vector
        flat = aligned.ravel()

        # Transform the feature vector using the pre-trained UMAP model
        # Note: flat[None, :] ensures the input has the correct shape (1, N)
        return self.embedding_model.transform(flat[None, :])[0]

    """Evaluator implementation for MAP-Elites."""

    def evaluate(self, sol):
        """Submits a solution to the external API for evaluation and computes descriptor/fitness."""
        sol_id = sol.get("id", 0)
        ok = True
        msg = ""
        desc = np.zeros((2,))  # Default descriptor
        fit_score = INVALID_SCORE

        try:
            # 1. Send solution for evaluation
            r = requests.post(f"{BASE_URL}/evaluate", json=sol, timeout=60)
            r_json = r.json()
            if not r.ok:
                raise Exception(f"API error {r.status_code}: {r_json.get('error', r.text)}")

            # 2. Extract raw fitness metrics and compute descriptor
            fit = r_json.get("fitness", {})
            desc = self.descriptor_from_track(r_json)

            # 3. Compute final fitness score
            fit_score = self.fitness_formula(fit)

        except Exception as e:
            ok = False
            msg = str(e)

        return sol_id, ok, msg, fit_score, desc


class EvaluatorNoveltySearch(Evaluator):
    def __init__(self,
                 model_path="mapelite/embeddings/models/model_metrics_VAE/model_metrics_VAE_latent32.pth"
                 ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model, self.embedding_dim = metrics_model.load_model(
            self.device, model_path)

    def descriptor_from_metrics(self, metrics):
        metrics = np.array(metrics, dtype=np.float32)
        metrics = metrics_model.preprocess_data(metrics)
        data_tensor = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, var = self.embedding_model.encode(data_tensor, None)

        return mu.cpu().numpy()[0]

    def evaluate(self, sol):
        sol_id = sol.get("id", 0)
        ok = True
        msg = ""
        desc = np.zeros((self.embedding_dim,))  # Default descriptor
        fit_score = INVALID_SCORE

        try:
            # 1. Send solution for evaluation
            r = requests.post(f"{BASE_URL}/evaluate", json=sol, timeout=60)
            r_json = r.json()
            if not r.ok:
                raise Exception(f"API error {r.status_code}: {r_json.get('error', r.text)}")

            # 2. Extract raw fitness metrics
            fit = r_json.get("fitness", {})
            print(f"Raw fitness metrics for solution {sol_id}: {fit}")
            
            desc = self.descriptor_from_metrics(fit.get("embedding_data", []))
            # 3. Compute final fitness score
            fit_score = self.fitness_formula(fit)

        except Exception as e:
            ok = False
            msg = str(e)

        return sol_id, ok, msg, fit_score, desc
