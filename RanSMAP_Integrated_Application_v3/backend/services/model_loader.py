
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


class RanSMAP_CNN_LSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_features = 23

        self.cnns = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=10
            )
            for _ in range(23)
        ])

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(
            input_size=23,
            hidden_size=256,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256, 2)

    def forward(self, x):

        features = []

        for i in range(23):
            feature = x[:, :, i].unsqueeze(1)
            feature = self.cnns[i](feature)
            feature = self.pool(feature)
            features.append(feature.squeeze(1))

        x = torch.stack(features, dim=2)

        x, _ = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)

        return self.fc(hidden[-1])

@lru_cache(maxsize=1)
def load_model():

    model_dir = Path(__file__).resolve().parents[1] / "models"

    model_path = model_dir / "threshold_model.pth.zip"
    mean_path = model_dir / "feature_mean.npy"
    std_path = model_dir / "feature_std.npy"
    threshold_path = model_dir / "selected_threshold.txt"

    device = torch.device("cpu")

    # Load trained weights
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model = RanSMAP_CNN_LSTM().to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load normalization values
    feature_mean = torch.tensor(
        np.load(mean_path),
        dtype=torch.float32
    )

    feature_std = torch.tensor(
        np.load(std_path),
        dtype=torch.float32
    )

    # Load selected threshold
    threshold = float(threshold_path.read_text().strip())

    if feature_mean.shape != (23,):
        raise ValueError("Feature mean must contain 23 values.")

    if feature_std.shape != (23,) or torch.any(feature_std <= 0):
        raise ValueError("Feature standard deviation is invalid.")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid prediction threshold.")

    return model, feature_mean, feature_std, threshold
