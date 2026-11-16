import numpy as np
import torch

from backend.services.baseline_model_loader import load_baseline_model


def predict_features(features):
    """Run the original seed-42 baseline using its own training statistics."""
    features = np.asarray(features, dtype=np.float64)

    if features.shape != (300, 23):
        raise ValueError(f"Expected (300, 23) features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity.")

    model, feature_mean, feature_std = load_baseline_model()
    normalized = (
        (features - feature_mean.reshape(1, 23))
        / feature_std.reshape(1, 23)
    ).astype(np.float32)
    input_tensor = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    benign_probability = float(probabilities[0, 0].item())
    ransomware_probability = float(probabilities[0, 1].item())

    # The original baseline uses argmax; ties go to class 0 (Benign).
    prediction = (
        "Ransomware"
        if ransomware_probability > benign_probability
        else "Benign"
    )

    return {
        "prediction": prediction,
        "benign_probability": benign_probability,
        "ransomware_probability": ransomware_probability,
        "threshold": 0.5,
    }
