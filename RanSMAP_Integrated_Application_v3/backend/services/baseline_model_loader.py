from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
import torch

from backend.services.model_loader import RanSMAP_CNN_LSTM


@lru_cache(maxsize=1)
def load_baseline_model():
    """Load the original seed-42 baseline and its training-fitted statistics."""
    model_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ransmap_cnn_lstm_original_td30_tw0p1.pkl"
    )

    with model_path.open("rb") as file_obj:
        bundle = pickle.load(file_obj)

    if not isinstance(bundle, dict) or bundle.get("input_shape") != (300, 23):
        raise ValueError("The original baseline bundle has an invalid format.")

    mean = np.asarray(bundle.get("feature_mean"), dtype=np.float64)
    std = np.asarray(bundle.get("feature_std"), dtype=np.float64)
    if mean.shape != (23,) or std.shape != (23,):
        raise ValueError("Baseline normalization arrays must have 23 values.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Baseline normalization arrays are invalid.")

    state = bundle.get("primary_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("The baseline bundle has no primary model weights.")

    state_tensors = {
        name: torch.as_tensor(np.asarray(value).copy())
        for name, value in state.items()
    }
    model = RanSMAP_CNN_LSTM()
    model.load_state_dict(state_tensors, strict=True)
    model.eval()

    return model, mean, std
