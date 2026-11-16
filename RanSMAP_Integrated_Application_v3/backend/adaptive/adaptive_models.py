"""PyTorch model/artifact boundary for the RanSMAP adaptive service.

Important:
    This module does NOT recreate the RanSMAP CNN-LSTM architecture.

The real RanSMAP architecture implementation and threshold-tuned .pth
checkpoint must be supplied separately. If the checkpoint contains only a
state_dict, an explicit model factory is required.

This module is responsible for:
    - normalization artifact validation
    - PyTorch checkpoint loading
    - model-factory resolution
    - inference tensor preparation
    - logits -> softmax conversion
    - model cloning
    - isolated candidate training
    - atomic checkpoint persistence
    - checkpoint metadata validation
    - explicit unavailable-artifact/device states

It does not implement adaptive-learning policy decisions.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
from torch import nn


INPUT_SHAPE = (300, 23)
NUM_CLASSES = 2

CLASS_BENIGN = 0
CLASS_RANSOMWARE = 1

CLASS_MAPPING = {
    CLASS_BENIGN: "Benign",
    CLASS_RANSOMWARE: "Ransomware",
}


class ArtifactLoadError(RuntimeError):
    """Raised when an artifact cannot be loaded safely."""


class ArtifactCompatibilityError(RuntimeError):
    """Raised when an artifact is structurally incompatible."""


class ModelUnavailableError(RuntimeError):
    """Raised when a required model artifact is unavailable."""


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested inference device is invalid/unavailable."""


class ModelTrainingError(RuntimeError):
    """Raised when candidate model training cannot be completed safely."""


class ModelRole(str, Enum):
    ACTIVE = "active"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class ModelArtifactState:
    checkpoint_identity: str
    model_version: Optional[str]
    role: ModelRole


@dataclass(frozen=True)
class ModelAvailability:
    available: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class NormalizationArtifacts:
    """
    Immutable normalization artifacts.

    The original arrays are validated before conversion to float64 so that
    complex values, strings, malformed objects, NaN, infinity, and incorrect
    shapes cannot be silently coerced into something apparently valid.
    """

    feature_mean: np.ndarray
    feature_std: np.ndarray

    def __post_init__(self) -> None:
        mean_original = np.asarray(self.feature_mean)
        std_original = np.asarray(self.feature_std)

        self._validate_original_array(
            mean_original,
            name="feature_mean",
        )
        self._validate_original_array(
            std_original,
            name="feature_std",
        )

        if mean_original.shape != (INPUT_SHAPE[1],):
            raise ArtifactCompatibilityError(
                f"feature_mean must have shape {(INPUT_SHAPE[1],)}, "
                f"got {mean_original.shape}"
            )

        if std_original.shape != (INPUT_SHAPE[1],):
            raise ArtifactCompatibilityError(
                f"feature_std must have shape {(INPUT_SHAPE[1],)}, "
                f"got {std_original.shape}"
            )

        mean = np.array(mean_original, dtype=np.float64, copy=True)
        std = np.array(std_original, dtype=np.float64, copy=True)

        if np.any(std <= 0):
            raise ArtifactCompatibilityError(
                "feature_std must contain strictly positive values"
            )

        mean.setflags(write=False)
        std.setflags(write=False)

        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_std", std)

    @staticmethod
    def _validate_original_array(
        array: np.ndarray,
        *,
        name: str,
    ) -> None:
        """
        Validate the original ndarray before any dtype conversion.

        In particular, complex values are rejected before astype(float64),
        preventing silent loss of imaginary components.
        """
        if np.iscomplexobj(array):
            raise ArtifactCompatibilityError(
                f"{name} must not contain complex-valued data"
            )

        if not np.issubdtype(array.dtype, np.number):
            raise ArtifactCompatibilityError(
                f"{name} must contain numeric values; "
                f"got dtype {array.dtype}"
            )

        if not np.all(np.isfinite(array)):
            raise ArtifactCompatibilityError(
                f"{name} must contain only finite values"
            )

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize a single RanSMAP sample with shape (300, 23)."""

        original = np.asarray(features)

        if np.iscomplexobj(original):
            raise ArtifactCompatibilityError(
                "Input features must not contain complex-valued data"
            )

        if not np.issubdtype(original.dtype, np.number):
            raise ArtifactCompatibilityError(
                f"Input features must be numeric; got dtype {original.dtype}"
            )

        if original.shape != INPUT_SHAPE:
            raise ArtifactCompatibilityError(
                f"Input features must have shape {INPUT_SHAPE}, "
                f"got {original.shape}"
            )

        if not np.all(np.isfinite(original)):
            raise ArtifactCompatibilityError(
                "Input features must contain only finite values"
            )

        x = np.array(original, dtype=np.float64, copy=True)

        normalized = (
            (x - self.feature_mean) / self.feature_std
        )

        if not np.all(np.isfinite(normalized)):
            raise ArtifactCompatibilityError(
                "Normalized input contains non-finite values"
            )

        return normalized


ModelFactory = Callable[[], nn.Module]


def resolve_architecture_loader(loader_spec: str) -> ModelFactory:
    """
    Resolve an architecture factory specified as:

        package.module:function

    The returned callable is expected to construct the actual RanSMAP
    architecture. This function does not invent or reconstruct that
    architecture.
    """
    if not isinstance(loader_spec, str) or not loader_spec.strip():
        raise ArtifactCompatibilityError(
            "Architecture loader specification must be a non-empty string"
        )

    if ":" not in loader_spec:
        raise ArtifactCompatibilityError(
            "Architecture loader must use 'package.module:function' format"
        )

    module_name, function_name = loader_spec.split(":", 1)

    if not module_name or not function_name:
        raise ArtifactCompatibilityError(
            "Architecture loader must use 'package.module:function' format"
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Unable to import architecture module '{module_name}'"
        ) from exc

    try:
        factory = getattr(module, function_name)
    except AttributeError as exc:
        raise ArtifactLoadError(
            f"Architecture callable '{function_name}' was not found in "
            f"module '{module_name}'"
        ) from exc

    if not callable(factory):
        raise ArtifactCompatibilityError(
            f"Architecture loader '{loader_spec}' is not callable"
        )

    def wrapped_factory() -> nn.Module:
        model = factory()

        if not isinstance(model, nn.Module):
            raise ArtifactCompatibilityError(
                "Architecture factory must return torch.nn.Module; "
                f"got {type(model).__name__}"
            )

        return model

    return wrapped_factory


def resolve_device(device_spec: str) -> torch.device:
    """Resolve and validate a PyTorch device specification."""

    if not isinstance(device_spec, str) or not device_spec.strip():
        raise DeviceUnavailableError(
            "Device specification must be a non-empty string"
        )

    try:
        device = torch.device(device_spec)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise DeviceUnavailableError(
            f"Invalid PyTorch device specification: {device_spec!r}"
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"CUDA requested but CUDA is unavailable: {device_spec}"
            )

        if device.index is not None:
            device_count = torch.cuda.device_count()
            if device.index >= device_count:
                raise DeviceUnavailableError(
                    f"CUDA device index {device.index} is unavailable; "
                    f"device count is {device_count}"
                )

    elif device.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise DeviceUnavailableError(
                "MPS requested but MPS is unavailable"
            )

    elif device.type != "cpu":
        raise DeviceUnavailableError(
            f"Unsupported inference device type: {device.type!r}"
        )

    return device


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """
    Extract a state_dict from common safe PyTorch checkpoint structures.

    Serialized complete nn.Module objects are deliberately rejected.
    """
    if isinstance(payload, nn.Module):
        raise ArtifactCompatibilityError(
            "Serialized nn.Module checkpoints are not accepted; "
            "provide a state_dict checkpoint and an explicit architecture factory"
        )

    if not isinstance(payload, Mapping):
        raise ArtifactCompatibilityError(
            "Checkpoint must contain a state_dict mapping"
        )

    if "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ArtifactCompatibilityError(
            "Checkpoint state_dict must be a non-empty mapping"
        )

    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ArtifactCompatibilityError(
                "Checkpoint state_dict keys must be strings"
            )

        if not isinstance(value, torch.Tensor):
            raise ArtifactCompatibilityError(
                f"Checkpoint value for '{key}' is not a torch.Tensor"
            )

    return state_dict


def load_pytorch_checkpoint(
    checkpoint_path: Path | str,
    *,
    model_factory: ModelFactory,
    device: str = "cpu",
) -> "LoadedModel":
    """
    Safely load a PyTorch state-dict checkpoint.

    The model_factory is validated here even when supplied directly by a
    caller, rather than through resolve_architecture_loader().
    """
    path = Path(checkpoint_path)

    if not path.is_file():
        raise ArtifactLoadError(
            f"Checkpoint file does not exist: {path}"
        )

    if not callable(model_factory):
        raise ArtifactCompatibilityError(
            "model_factory must be callable"
        )

    resolved_device = resolve_device(device)

    try:
        payload = torch.load(
            path,
            map_location=resolved_device,
            weights_only=True,
        )
    except TypeError as exc:
        raise ArtifactLoadError(
            "The installed PyTorch version does not support the required "
            "weights_only=True checkpoint loading mode"
        ) from exc
    except Exception as exc:
        raise ArtifactLoadError(
            f"Unable to safely load checkpoint: {path}"
        ) from exc

    state_dict = _extract_state_dict(payload)

    try:
        model = model_factory()
    except Exception as exc:
        raise ArtifactLoadError(
            "Architecture factory failed while constructing the model"
        ) from exc

    if not isinstance(model, nn.Module):
        raise ArtifactCompatibilityError(
            "model_factory must return torch.nn.Module; "
            f"got {type(model).__name__}"
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ArtifactCompatibilityError(
            "Checkpoint state_dict is incompatible with the supplied "
            "architecture"
        ) from exc

    model.to(resolved_device)
    model.eval()

    return LoadedModel(
        model=model,
        normalization=None,
        device=resolved_device,
    )


def _infer_model_dtype(model: nn.Module) -> torch.dtype:
    """Infer the floating dtype expected by the loaded model."""

    for parameter in model.parameters():
        if parameter.dtype.is_floating_point:
            return parameter.dtype

    for buffer in model.buffers():
        if buffer.dtype.is_floating_point:
            return buffer.dtype

    return torch.float32


@dataclass
class LoadedModel:
    """
    Runtime wrapper around a loaded PyTorch model and normalization artifacts.
    """

    model: nn.Module
    normalization: Optional[NormalizationArtifacts]
    device: torch.device
    artifact_state: Optional[ModelArtifactState] = None

    @property
    def model_version(self) -> Optional[str]:
        """Return the configured model version, if available."""

        if self.artifact_state is None:
            return None

        return self.artifact_state.model_version

    def predict_logits(self, features: np.ndarray) -> torch.Tensor:
        """Return raw model logits as a detached CPU tensor of shape (1, 2)."""

        if self.normalization is None:
            raise ModelUnavailableError(
                "Normalization artifacts are not configured"
            )

        normalized = self.normalization.normalize(features)

        dtype = _infer_model_dtype(self.model)

        tensor = torch.as_tensor(
            normalized,
            dtype=dtype,
            device=self.device,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            logits = self.model(tensor)

        if not isinstance(logits, torch.Tensor):
            raise ArtifactCompatibilityError(
                "Model forward pass must return a torch.Tensor"
            )

        if logits.shape != (1, NUM_CLASSES):
            raise ArtifactCompatibilityError(
                f"Model logits must have shape (1, {NUM_CLASSES}), "
                f"got {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise ArtifactCompatibilityError(
                "Model produced non-finite logits"
            )

        return logits.detach().cpu()

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Convert raw logits to probabilities using torch.softmax.

        Class order:
            index 0 -> Benign
            index 1 -> Ransomware
        """
        logits = self.predict_logits(features)

        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)

        probabilities = torch.softmax(logits_tensor, dim=1)

        if probabilities.shape != (1, NUM_CLASSES):
            raise ArtifactCompatibilityError(
                f"Probabilities must have shape (1, {NUM_CLASSES}), "
                f"got {tuple(probabilities.shape)}"
            )

        if not torch.isfinite(probabilities).all():
            raise ArtifactCompatibilityError(
                "Model produced non-finite probabilities"
            )

        return probabilities.squeeze(0).numpy()

    def predict_ransomware_probability(
        self,
        features: np.ndarray,
    ) -> float:
        """Return class-1 ransomware probability."""

        probabilities = self.predict_probabilities(features)

        return float(probabilities[CLASS_RANSOMWARE])


def clone_model(
    loaded_model: LoadedModel,
    *,
    role: ModelRole = ModelRole.CANDIDATE,
) -> LoadedModel:
    """
    Deep-copy the model parameters.

    Normalization artifacts remain shared intentionally because they are
    immutable/read-only after construction.
    """
    cloned_nn = copy.deepcopy(loaded_model.model)

    cloned_state = loaded_model.artifact_state

    if cloned_state is not None:
        cloned_state = ModelArtifactState(
            checkpoint_identity=cloned_state.checkpoint_identity,
            model_version=cloned_state.model_version,
            role=role,
        )

    return LoadedModel(
        model=cloned_nn,
        normalization=loaded_model.normalization,
        device=loaded_model.device,
        artifact_state=cloned_state,
    )


@dataclass(frozen=True)
class TrainingReport:
    """Structured result of one isolated candidate-training run."""

    num_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    initial_loss: float
    final_loss: float
    training_accuracy: float
    parameters_changed: bool
    success: bool
    message: str


def _validate_training_features(features: Any) -> np.ndarray:
    """Validate raw training features without silently reshaping or coercing."""

    if not isinstance(features, np.ndarray):
        raise ModelTrainingError(
            f"Training features must be a numpy.ndarray; "
            f"got {type(features).__name__}"
        )

    if np.iscomplexobj(features):
        raise ModelTrainingError(
            "Training features must not contain complex-valued data"
        )

    if not np.issubdtype(features.dtype, np.number):
        raise ModelTrainingError(
            f"Training features must contain numeric values; "
            f"got dtype {features.dtype}"
        )

    if features.ndim != 3 or features.shape[1:] != INPUT_SHAPE:
        raise ModelTrainingError(
            f"Training features must have shape "
            f"(N, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}), "
            f"got {features.shape}"
        )

    if features.shape[0] <= 0:
        raise ModelTrainingError(
            "Training features must contain at least one sample"
        )

    if not np.all(np.isfinite(features)):
        raise ModelTrainingError(
            "Training features must contain only finite values"
        )

    return np.array(features, dtype=np.float64, copy=True)


def _validate_training_labels(
    labels: Any,
    num_samples: int,
) -> np.ndarray:
    """Validate binary integer class IDs without silently changing label values."""

    if not isinstance(labels, np.ndarray):
        raise ModelTrainingError(
            f"Training labels must be a numpy.ndarray; "
            f"got {type(labels).__name__}"
        )

    if np.iscomplexobj(labels):
        raise ModelTrainingError(
            "Training labels must not contain complex-valued data"
        )

    if labels.ndim != 1:
        raise ModelTrainingError(
            f"Training labels must have shape (N,); got {labels.shape}"
        )

    if labels.shape[0] != num_samples:
        raise ModelTrainingError(
            f"Number of labels ({labels.shape[0]}) must equal "
            f"number of samples ({num_samples})"
        )

    if not np.issubdtype(labels.dtype, np.integer):
        raise ModelTrainingError(
            f"Training labels must contain integer class IDs; "
            f"got dtype {labels.dtype}"
        )

    if not np.all(
        np.isin(labels, (CLASS_BENIGN, CLASS_RANSOMWARE))
    ):
        raise ModelTrainingError(
            "Training labels must contain only class IDs 0 or 1"
        )

    return np.array(labels, dtype=np.int64, copy=True)


def _validate_training_parameters(
    learning_rate: Any,
    optimizer_name: Any,
    epochs: Any,
    batch_size: Any,
) -> tuple[float, str, int, int]:
    """Validate explicit training parameters."""

    if (
        isinstance(learning_rate, bool)
        or not isinstance(
            learning_rate,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ModelTrainingError(
            "learning_rate must be a positive finite number"
        )

    learning_rate_value = float(learning_rate)

    if (
        not math.isfinite(learning_rate_value)
        or learning_rate_value <= 0
    ):
        raise ModelTrainingError(
            "learning_rate must be a positive finite number"
        )

    if (
        not isinstance(optimizer_name, str)
        or not optimizer_name.strip()
    ):
        raise ModelTrainingError(
            "optimizer_name must be a non-empty string"
        )

    optimizer_key = optimizer_name.strip().lower()

    if optimizer_key not in {"adam", "sgd"}:
        raise ModelTrainingError(
            f"Unsupported optimizer '{optimizer_name}'. "
            "Supported optimizers: Adam, SGD"
        )

    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, np.integer))
    ):
        raise ModelTrainingError(
            "epochs must be a positive integer"
        )

    epochs_value = int(epochs)

    if epochs_value <= 0:
        raise ModelTrainingError(
            "epochs must be a positive integer"
        )

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
    ):
        raise ModelTrainingError(
            "batch_size must be a positive integer"
        )

    batch_size_value = int(batch_size)

    if batch_size_value <= 0:
        raise ModelTrainingError(
            "batch_size must be a positive integer"
        )

    return (
        learning_rate_value,
        optimizer_key,
        epochs_value,
        batch_size_value,
    )


def _snapshot_model_parameters(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Capture detached CPU copies for post-training comparison."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _parameters_changed(
    before: Mapping[str, torch.Tensor],
    after_model: nn.Module,
) -> bool:
    """Return whether any candidate parameter differs from its snapshot."""

    after = dict(after_model.named_parameters())

    if set(before) != set(after):
        raise ModelTrainingError(
            "Model parameter structure changed during candidate training"
        )

    return any(
        not torch.equal(
            before[name],
            parameter.detach().cpu(),
        )
        for name, parameter in after.items()
    )


def _ensure_finite_model_parameters(
    model: nn.Module,
    *,
    stage: str,
) -> None:
    """Reject non-finite trainable parameters."""

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            raise ModelTrainingError(
                f"Model parameter '{name}' became non-finite "
                f"during {stage}"
            )


def train_candidate_model(
    loaded_model: LoadedModel,
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    optimizer_name: str,
    epochs: int,
    batch_size: int,
) -> tuple[LoadedModel, TrainingReport]:
    """
    Train an isolated candidate model from labelled raw RanSMAP samples.

    The supplied LoadedModel is treated as the active/source model and is
    never trained, switched to training mode, or otherwise modified.

    Training occurs on a deep clone of the supplied model and uses the
    supplied model's immutable normalization artifacts.

    Training parameters are intentionally explicit. No adaptive-policy
    defaults are introduced here.

    Returns:
        A separately trained candidate LoadedModel and a TrainingReport.

    Raises:
        ModelTrainingError:
            If validation or training fails.
        DeviceUnavailableError:
            If the configured training device is unavailable.
    """

    if not isinstance(loaded_model, LoadedModel):
        raise ModelTrainingError(
            f"loaded_model must be a LoadedModel; "
            f"got {type(loaded_model).__name__}"
        )

    if loaded_model.normalization is None:
        raise ModelTrainingError(
            "Normalization artifacts are unavailable; "
            "candidate training cannot proceed"
        )

    if not isinstance(loaded_model.model, nn.Module):
        raise ModelTrainingError(
            "loaded_model.model must be a torch.nn.Module"
        )

    if not isinstance(loaded_model.device, torch.device):
        raise ModelTrainingError(
            "loaded_model.device must be a torch.device"
        )

    if (
        loaded_model.device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise DeviceUnavailableError(
            "Candidate training requested CUDA but CUDA is unavailable"
        )

    raw_features = _validate_training_features(features)
    raw_labels = _validate_training_labels(
        labels,
        raw_features.shape[0],
    )

    (
        learning_rate_value,
        optimizer_key,
        epochs_value,
        batch_size_value,
    ) = _validate_training_parameters(
        learning_rate,
        optimizer_name,
        epochs,
        batch_size,
    )

    # Clone before any candidate-side state change. The active/source model
    # is never passed to the optimizer.
    candidate = clone_model(
        loaded_model,
        role=ModelRole.CANDIDATE,
    )

    candidate.model.to(candidate.device)
    candidate.model.eval()

    active_parameter_snapshot = _snapshot_model_parameters(
        loaded_model.model
    )

    # Normalize the raw training samples exactly once using the existing
    # training-fitted artifacts.
    normalized_features = np.stack(
        [
            loaded_model.normalization.normalize(sample)
            for sample in raw_features
        ],
        axis=0,
    )

    if not np.all(np.isfinite(normalized_features)):
        raise ModelTrainingError(
            "Normalized training features contain non-finite values"
        )

    dtype = _infer_model_dtype(candidate.model)

    training_tensor = torch.as_tensor(
        normalized_features,
        dtype=dtype,
        device=candidate.device,
    )

    label_tensor = torch.as_tensor(
        raw_labels,
        dtype=torch.long,
        device=candidate.device,
    )

    if training_tensor.shape != (
        raw_features.shape[0],
        *INPUT_SHAPE,
    ):
        raise ModelTrainingError(
            "Normalized training tensor has unexpected shape "
            f"{tuple(training_tensor.shape)}"
        )

    if label_tensor.shape != (raw_features.shape[0],):
        raise ModelTrainingError(
            "Training label tensor has unexpected shape "
            f"{tuple(label_tensor.shape)}"
        )

    if optimizer_key == "adam":
        optimizer = torch.optim.Adam(
            candidate.model.parameters(),
            lr=learning_rate_value,
        )
        optimizer_display_name = "Adam"
    else:
        optimizer = torch.optim.SGD(
            candidate.model.parameters(),
            lr=learning_rate_value,
        )
        optimizer_display_name = "SGD"

    criterion = nn.CrossEntropyLoss()

    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None

    try:
        # Measure the complete training dataset before optimization so the
        # initial and final losses use the same evaluation procedure.
        candidate.model.eval()

        with torch.no_grad():
            initial_logits = candidate.model(training_tensor)

            if not isinstance(initial_logits, torch.Tensor):
                raise ModelTrainingError(
                    "Model forward pass must return a torch.Tensor"
                )

            if tuple(initial_logits.shape) != (
                raw_features.shape[0],
                NUM_CLASSES,
            ):
                raise ModelTrainingError(
                    "Initial evaluation logits have an unexpected shape"
                )

            if not torch.isfinite(initial_logits).all():
                raise ModelTrainingError(
                    "Initial evaluation logits are non-finite"
                )

            initial_loss = float(
                criterion(
                    initial_logits,
                    label_tensor,
                )
                .detach()
                .cpu()
                .item()
            )

        candidate.model.train()

        for epoch_index in range(epochs_value):
            running_loss = 0.0
            seen = 0

            for start in range(
                0,
                raw_features.shape[0],
                batch_size_value,
            ):
                end = min(
                    start + batch_size_value,
                    raw_features.shape[0],
                )

                batch_features = training_tensor[start:end]
                batch_labels = label_tensor[start:end]

                optimizer.zero_grad(set_to_none=True)

                logits = candidate.model(batch_features)

                if not isinstance(logits, torch.Tensor):
                    raise ModelTrainingError(
                        "Model forward pass must return a torch.Tensor"
                    )

                expected_shape = (
                    end - start,
                    NUM_CLASSES,
                )

                if tuple(logits.shape) != expected_shape:
                    raise ModelTrainingError(
                        f"Model logits must have shape {expected_shape}; "
                        f"got {tuple(logits.shape)}"
                    )

                if not torch.isfinite(logits).all():
                    raise ModelTrainingError(
                        f"Non-finite logits encountered during "
                        f"epoch {epoch_index + 1}"
                    )

                loss = criterion(
                    logits,
                    batch_labels,
                )

                if not torch.isfinite(loss):
                    raise ModelTrainingError(
                        f"Non-finite loss encountered during "
                        f"epoch {epoch_index + 1}"
                    )

                loss.backward()

                for name, parameter in candidate.model.named_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(
                            parameter.grad
                        ).all()
                    ):
                        raise ModelTrainingError(
                            f"Non-finite gradient encountered for "
                            f"parameter '{name}' during epoch "
                            f"{epoch_index + 1}"
                        )

                optimizer.step()

                _ensure_finite_model_parameters(
                    candidate.model,
                    stage=f"epoch {epoch_index + 1}",
                )

                batch_count = end - start

                running_loss += (
                    float(loss.detach().cpu().item())
                    * batch_count
                )

                seen += batch_count

            if seen != raw_features.shape[0]:
                raise ModelTrainingError(
                    f"Training processed {seen} samples but expected "
                    f"{raw_features.shape[0]}"
                )

            epoch_loss = running_loss / seen

            if not math.isfinite(epoch_loss):
                raise ModelTrainingError(
                    f"Non-finite epoch loss encountered during "
                    f"epoch {epoch_index + 1}"
                )

        candidate.model.eval()

        # Evaluate the complete training dataset after optimization. This
        # final loss is directly comparable with initial_loss above.
        with torch.no_grad():
            evaluation_logits = candidate.model(
                training_tensor
            )

            if not isinstance(
                evaluation_logits,
                torch.Tensor,
            ):
                raise ModelTrainingError(
                    "Model forward pass must return a torch.Tensor"
                )

            if tuple(evaluation_logits.shape) != (
                raw_features.shape[0],
                NUM_CLASSES,
            ):
                raise ModelTrainingError(
                    "Final evaluation logits have an unexpected shape"
                )

            if not torch.isfinite(
                evaluation_logits
            ).all():
                raise ModelTrainingError(
                    "Final evaluation logits are non-finite"
                )

            final_loss = float(
                criterion(
                    evaluation_logits,
                    label_tensor,
                )
                .detach()
                .cpu()
                .item()
            )

            if not math.isfinite(final_loss):
                raise ModelTrainingError(
                    "Final evaluation loss is non-finite"
                )

            training_accuracy = float(
                (
                    evaluation_logits.argmax(dim=1)
                    == label_tensor
                )
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )

        if initial_loss is None or final_loss is None:
            raise ModelTrainingError(
                "Training completed without producing comparable loss values"
            )

        candidate_parameters_changed = _parameters_changed(
            active_parameter_snapshot,
            candidate.model,
        )

        source_changed = any(
            not torch.equal(
                before,
                parameter.detach().cpu(),
            )
            for name, before in active_parameter_snapshot.items()
            for parameter_name, parameter
            in loaded_model.model.named_parameters()
            if name == parameter_name
        )

        if source_changed:
            raise ModelTrainingError(
                "Active/source model parameters changed during "
                "candidate training"
            )

        report = TrainingReport(
            num_samples=raw_features.shape[0],
            epochs=epochs_value,
            batch_size=batch_size_value,
            learning_rate=learning_rate_value,
            optimizer=optimizer_display_name,
            initial_loss=initial_loss,
            final_loss=final_loss,
            training_accuracy=training_accuracy,
            parameters_changed=candidate_parameters_changed,
            success=True,
            message="Candidate training completed successfully",
        )

        return candidate, report

    except Exception as exc:
        # The active model was never handed to the optimizer and was never
        # switched to training mode. The candidate is returned nowhere after
        # a failure, and its mode is reset before the exception leaves.
        candidate.model.eval()

        if isinstance(exc, ModelTrainingError):
            raise

        raise ModelTrainingError(
            f"Candidate training failed safely: {exc}"
        ) from exc


_RESERVED_METADATA_KEYS = frozenset(
    {
        "checkpoint_identity",
        "model_version",
        "model_role",
        "framework",
        "input_shape",
        "num_classes",
        "class_mapping",
    }
)


def _validate_json_compatible_metadata(
    value: Any,
    *,
    path: str = "extra_metadata",
) -> None:
    """
    Restrict metadata to values that can safely survive JSON serialization.

    Allowed:
        None
        native Python bool
        native Python int
        native Python finite float
        native Python str
        list/tuple of allowed values
        dict with string keys and allowed values

    Unsupported Python/NumPy scalar objects are rejected.
    """
    if value is None:
        return

    if type(value) in (bool, str, int):
        return

    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactCompatibilityError(
                f"{path} contains a non-finite float"
            )
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_compatible_metadata(
                item,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactCompatibilityError(
                    f"{path} contains a non-string dictionary key: {key!r}"
                )

            _validate_json_compatible_metadata(
                item,
                path=f"{path}.{key}",
            )
        return

    raise ArtifactCompatibilityError(
        f"{path} contains unsupported metadata type: "
        f"{type(value).__name__}"
    )

def _validate_extra_metadata(
    extra_metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and return a safe copy of additional checkpoint metadata."""

    if extra_metadata is None:
        return {}

    if not isinstance(extra_metadata, Mapping):
        raise ArtifactCompatibilityError(
            "extra_metadata must be a mapping"
        )

    conflicts = _RESERVED_METADATA_KEYS.intersection(
        extra_metadata.keys()
    )

    if conflicts:
        raise ArtifactCompatibilityError(
            "extra_metadata contains reserved metadata keys: "
            + ", ".join(sorted(conflicts))
        )

    copied = dict(extra_metadata)

    _validate_json_compatible_metadata(copied)

    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "extra_metadata is not JSON-compatible"
        ) from exc

    return copied

def save_checkpoint_atomic(
    loaded_model: LoadedModel,
    checkpoint_path: Path | str,
    *,
    checkpoint_identity: Optional[str] = None,
    model_version: Optional[str] = None,
    model_role: Optional[ModelRole] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint using same-directory temporary-file replacement.

    Atomic replacement protects readers from observing a partially written
    target file.

    It does NOT validate whether a candidate is worthy of becoming active.
    Candidate validation/acceptance and active-path promotion belong to the
    adaptive service lifecycle. Candidate and active checkpoint paths must
    therefore remain distinct.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _validate_extra_metadata(extra_metadata)

    state = loaded_model.artifact_state

    resolved_identity = (
        checkpoint_identity
        if checkpoint_identity is not None
        else state.checkpoint_identity
        if state is not None
        else None
    )

    resolved_version = (
        model_version
        if model_version is not None
        else state.model_version
        if state is not None
        else None
    )

    resolved_role = (
        model_role
        if model_role is not None
        else state.role
        if state is not None
        else None
    )

    if not resolved_identity:
        raise ArtifactCompatibilityError(
            "checkpoint_identity must be explicitly configured"
        )

    if resolved_role is None:
        raise ArtifactCompatibilityError(
            "model_role must be explicitly configured"
        )

    complete_metadata = {
        "checkpoint_identity": resolved_identity,
        "model_version": resolved_version,
        "model_role": resolved_role.value,
        "framework": "PyTorch",
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "class_mapping": {
            "0": "Benign",
            "1": "Ransomware",
        },
        **metadata,
    }

    _validate_json_compatible_metadata(complete_metadata)

    payload = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value
            in loaded_model.model.state_dict().items()
        },
        "metadata": complete_metadata,
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "wb") as temporary_file:
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_name, path)

    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_saved_checkpoint_metadata(
    checkpoint_path: Path | str,
) -> dict[str, Any]:
    """
    Safely reload checkpoint metadata using weights_only=True.

    This is intentionally limited to metadata inspection and does not
    reconstruct the actual model architecture.
    """
    path = Path(checkpoint_path)

    if not path.is_file():
        raise ArtifactLoadError(
            f"Checkpoint file does not exist: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise ArtifactLoadError(
            "The installed PyTorch version does not support "
            "weights_only=True"
        ) from exc
    except Exception as exc:
        raise ArtifactLoadError(
            f"Unable to safely reload checkpoint: {path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ArtifactCompatibilityError(
            "Saved checkpoint payload is not a mapping"
        )

    metadata = payload.get("metadata")

    if not isinstance(metadata, Mapping):
        raise ArtifactCompatibilityError(
            "Saved checkpoint does not contain a metadata mapping"
        )

    return dict(metadata)

def initialize_model(
    config: Any,
    *,
    model_factory: Optional[ModelFactory] = None,
) -> LoadedModel:
    """
    Initialize the adaptive integration model.

    Required artifacts:
        - threshold-tuned integration checkpoint
        - matching feature_mean.npy
        - matching feature_std.npy

    There is deliberately no fallback to the original random-state baseline
    checkpoint.
    """
    artifacts = config.artifacts

    checkpoint_path = artifacts.integration_checkpoint
    mean_path = artifacts.feature_mean
    std_path = artifacts.feature_std

    if not checkpoint_path:
        raise ModelUnavailableError(
            "Integration checkpoint path is not configured"
        )

    if not mean_path:
        raise ModelUnavailableError(
            "feature_mean path is not configured"
        )

    if not std_path:
        raise ModelUnavailableError(
            "feature_std path is not configured"
        )

    factory = model_factory

    if factory is None:
        loader_spec = config.model.architecture_loader

        if not loader_spec:
            raise ModelUnavailableError(
                "No architecture factory is configured"
            )

        factory = resolve_architecture_loader(loader_spec)

    device = resolve_device(config.model.device)

    try:
        mean = np.load(
            mean_path,
            allow_pickle=False,
        )
    except Exception as exc:
        raise ArtifactLoadError(
            f"Unable to load feature_mean artifact: {mean_path}"
        ) from exc

    try:
        std = np.load(
            std_path,
            allow_pickle=False,
        )
    except Exception as exc:
        raise ArtifactLoadError(
            f"Unable to load feature_std artifact: {std_path}"
        ) from exc

    normalization = NormalizationArtifacts(
        feature_mean=mean,
        feature_std=std,
    )

    loaded = load_pytorch_checkpoint(
        checkpoint_path,
        model_factory=factory,
        device=str(device),
    )

    loaded.normalization = normalization

    configured_identity = getattr(
        config.model,
        "checkpoint_identity",
        None,
    )

    checkpoint_identity = (
        configured_identity
        if configured_identity
        else "threshold_tuned_cnn_lstm_integration_checkpoint"
    )

    configured_version = getattr(
        config.model,
        "model_version",
        None,
    )

    loaded.artifact_state = ModelArtifactState(
        checkpoint_identity=checkpoint_identity,
        model_version=configured_version,
        role=ModelRole.ACTIVE,
    )

    return loaded

def unavailable_state(reason: str) -> ModelAvailability:
    """Represent an explicitly unavailable model state."""

    return ModelAvailability(
        available=False,
        reason=reason,
    )