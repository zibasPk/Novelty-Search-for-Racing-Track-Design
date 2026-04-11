import os

import numpy as np
import requests
import joblib
from mapelite.logging_config import get_logger
import torch

from mapelite.vae import MetricsTransformerVAE, MetricsPreprocessor
from mapelite.utils import EMBEDDING_MODEL, pca_align, solution_to_array, is_valid_solution_array
from abc import ABC, abstractmethod
from mapelite.config import (
    BASE_URL, INVALID_SCORE
)

log = get_logger(__name__)


class Evaluator(ABC):
    """Abstract base class for solution evaluators."""
    @abstractmethod
    def evaluate(self, sol):
        """Evaluates a solution."""
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
            log.warning("UMAP model not found — using placeholder", path=model_path)
            # Placeholder class to prevent crash if model isn't trained/found

            class PlaceholderUMAP:
                def transform(self, data):
                    # Returns a dummy 2D measure
                    return np.zeros((data.shape[0], 2))
            self.embedding_model = PlaceholderUMAP()

    def measure_from_track(self, sol):
        """Converts the track's spline vector into the 2D behavioral measure using UMAP."""
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
        """Submits a solution to the external API for evaluation and computes measure/fitness."""
        sol_id = sol.get("id", 0)
        ok = True
        msg = ""
        measure = np.zeros((2,))  # Default measure
        fit_score = INVALID_SCORE

        try:
            # 1. Send solution for evaluation
            r = requests.post(f"{BASE_URL}/evaluate", json=sol, timeout=60)
            r_json = r.json()
            if not r.ok:
                raise Exception(f"API error {r.status_code}: {r_json.get('error', r.text)}")

            # 2. Extract raw fitness metrics and compute measure
            fit = r_json.get("fitness", {})
            measure = self.measure_from_track(r_json)

            # 3. Compute final fitness score
            fit_score = self.fitness_formula(fit)

        except Exception as e:
            ok = False
            msg = str(e)

        return sol_id, ok, msg, fit_score, measure


class EvaluatorMetrics(Evaluator):
    def __init__(self, embedding_model, embedding_dim, device):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.device = device
        self.preprocessor = MetricsPreprocessor()
        # Set the model to evaluation mode
        self.embedding_model.eval()

        
    @classmethod
    def load_pretrained(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model, embedding_dim = MetricsTransformerVAE.load_pretrained(path, device)
        
        return cls(embedding_model, embedding_dim, device)

    @staticmethod
    def validate_metrics(metrics):
        """Checks if metrics are valid"""
        
        metrics = np.array(metrics, dtype=np.float32)
        # check the number of 0s in the steering is more thant 80% of the values, if so return False
        steering_metrics = metrics[:, 2]

        
        def calculate_zero_ratio(data):
            zero_count = np.sum(data == 0)
            total_count = len(data)
            if total_count == 0:
                return 1.0
            return zero_count / total_count
        
        steering_zero_ratio = calculate_zero_ratio(steering_metrics)
        if steering_zero_ratio > 0.8:
            log.debug(f"Steering metrics have {steering_zero_ratio:.2%} zeros, which is above the threshold.")
            return False
    
        return True
    
    def measure_from_metrics(self, metrics):
        metrics = np.array(metrics, dtype=np.float32)
        metrics = self.preprocessor(metrics)
        data_tensor = torch.tensor(metrics, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, var = self.embedding_model.encode(data_tensor, None)

        return mu.cpu().numpy()[0]

    def evaluate(self, sol):
        sol_id = sol.get("id", 0)
        ok = True
        msg = ""
        measure = np.zeros((self.embedding_dim,))  # Default measure
        fit_score = INVALID_SCORE
        phenotype_data = None

        if not is_valid_solution_array(solution_to_array(sol)):
            return sol_id, False, "Invalid solution array", fit_score, measure, phenotype_data

        try:
            r = requests.post(f"{BASE_URL}/evaluate", json=sol, timeout=60)
            r_json = r.json()
            if not r.ok:
                raise Exception(f"API error {r.status_code}: {r_json.get('error', r.text)}")

            # Extract raw fitness metrics
            fit = r_json.get("fitness", {})
            log.debug("Raw fitness metrics", sol_id=sol_id, metrics=fit)
            
            # Extract the metrics used for calculating the measure (embedding)
            phenotype_data = fit.get("embedding_data", [])
            
            if not EvaluatorMetrics.validate_metrics(phenotype_data):
                raise ValueError("Invalid metrics for embedding (too many zeros)")
            
            measure = self.measure_from_metrics(phenotype_data)
            
            # Compute final fitness score
            fit_score = self.fitness_formula(fit)

        except Exception as e:
            ok = False
            msg = str(e)

        return sol_id, ok, msg, fit_score, measure, phenotype_data
    
   