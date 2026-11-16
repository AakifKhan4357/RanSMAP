"""
Configuration for the RanSMAP adaptive backend. 
This module contains configuration only. Adaptive-learning implementation
logic belongs in adaptive_service.py and related helper modules.
Policy values that have not yet been finalized remain explicitly unresolved
rather than being assigned invented defaults.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Confirmed integration constants
# ---------------------------------------------------------------------------

INPUT_SHAPE = (300, 23)

CLASS_BENIGN = 0
CLASS_RANSOMWARE = 1

RANSOMWARE_THRESHOLD = 0.30

NORMALIZATION_FORMULA = (
    "(X - feature_mean) / feature_std"
)


# These identities are deliberately distinct.
#
# The threshold-tuned integration checkpoint is the artifact selected for
# adaptive integration. The original random-state-42 baseline checkpoint is
# retained as a distinct baseline identity and must not be used as an
# implicit fallback for adaptive initialization.

THRESHOLD_TUNED_INTEGRATION_CHECKPOINT = (
    "threshold_tuned_cnn_lstm_integration_checkpoint"
)

RANDOM_STATE_42_BASELINE_CHECKPOINT = (
    "original_random_state_42_baseline_checkpoint"
)


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactPaths:
    """
    Paths to model and normalization artifacts.

    The adaptive service requires the threshold-tuned integration checkpoint
    and its matching normalization arrays.
    """

    integration_checkpoint: Optional[Path] = None
    feature_mean: Optional[Path] = None
    feature_std: Optional[Path] = None

    # Kept separately for baseline/reference use. This is not an adaptive
    # fallback artifact.
    baseline_random_state_42_checkpoint: Optional[Path] = None


# ---------------------------------------------------------------------------
# Adaptive policy configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptivePolicyConfig:
    """
    Adaptive-learning policy settings.

    Some policy decisions remain unresolved. They are represented by None
    rather than invented implementation values.
    """

    # Drift detection
    drift_formula: Optional[str] = None
    drift_threshold: Optional[float] = None
    drift_monitoring_window: Optional[int] = None

    # Warning/replay buffers
    warning_buffer_capacity: Optional[int] = None
    replay_buffer_capacity: Optional[int] = None

    # Update triggering
    samples_required_before_update: Optional[int] = None

    # Replay/adaptation sampling
    replay_ratio: Optional[float] = None
    class_balance_strategy: Optional[str] = None

    # Candidate training
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None

    # Candidate acceptance
    candidate_acceptance_criteria: Optional[str] = None

    # Retention evaluation
    retention_evaluation_definition: Optional[str] = None


# ---------------------------------------------------------------------------
# Model identity configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelIdentityConfig:
    """
    Identity and runtime configuration for the adaptive model.

    The architecture_loader must resolve to an explicit architecture factory
    when loading a state-dict checkpoint. The current adaptive model loader
    does not accept complete serialized nn.Module checkpoints.
    """

    architecture_loader: Optional[str] = None

    selected_checkpoint_identity: str = (
        THRESHOLD_TUNED_INTEGRATION_CHECKPOINT
    )

    model_version: Optional[str] = None

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Complete adaptive configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveConfig:
    """
    Complete adaptive backend configuration.
    """

    artifacts: ArtifactPaths
    model: ModelIdentityConfig
    policy: AdaptivePolicyConfig

    @classmethod
    def from_environment(cls) -> "AdaptiveConfig":
        """
        Construct configuration from environment variables.

        Environment variables are intentionally limited to configuration
        values already defined by this module.
        """

        artifacts = ArtifactPaths(
            integration_checkpoint=_optional_path_from_env(
                "RANSMAP_INTEGRATION_CHECKPOINT"
            ),
            feature_mean=_optional_path_from_env(
                "RANSMAP_FEATURE_MEAN"
            ),
            feature_std=_optional_path_from_env(
                "RANSMAP_FEATURE_STD"
            ),
            baseline_random_state_42_checkpoint=_optional_path_from_env(
                "RANSMAP_BASELINE_RANDOM_STATE_42_CHECKPOINT"
            ),
        )

        model = ModelIdentityConfig(
            architecture_loader=os.getenv(
                "RANSMAP_ARCHITECTURE_LOADER"
            ),
            selected_checkpoint_identity=os.getenv(
                "RANSMAP_SELECTED_CHECKPOINT_IDENTITY",
                THRESHOLD_TUNED_INTEGRATION_CHECKPOINT,
            ),
            model_version=os.getenv(
                "RANSMAP_MODEL_VERSION"
            ),
            device=os.getenv(
                "RANSMAP_DEVICE",
                "cpu",
            ),
        )

        policy = AdaptivePolicyConfig(
            drift_formula=os.getenv(
                "RANSMAP_DRIFT_FORMULA"
            ),
            drift_threshold=_optional_float_from_env(
                "RANSMAP_DRIFT_THRESHOLD"
            ),
            drift_monitoring_window=_optional_int_from_env(
                "RANSMAP_DRIFT_MONITORING_WINDOW"
            ),
            warning_buffer_capacity=_optional_int_from_env(
                "RANSMAP_WARNING_BUFFER_CAPACITY"
            ),
            replay_buffer_capacity=_optional_int_from_env(
                "RANSMAP_REPLAY_BUFFER_CAPACITY"
            ),
            samples_required_before_update=_optional_int_from_env(
                "RANSMAP_SAMPLES_REQUIRED_BEFORE_UPDATE"
            ),
            replay_ratio=_optional_float_from_env(
                "RANSMAP_REPLAY_RATIO"
            ),
            class_balance_strategy=os.getenv(
                "RANSMAP_CLASS_BALANCE_STRATEGY"
            ),
            learning_rate=_optional_float_from_env(
                "RANSMAP_LEARNING_RATE"
            ),
            optimizer=os.getenv(
                "RANSMAP_OPTIMIZER"
            ),
            epochs=_optional_int_from_env(
                "RANSMAP_EPOCHS"
            ),
            batch_size=_optional_int_from_env(
                "RANSMAP_BATCH_SIZE"
            ),
            candidate_acceptance_criteria=os.getenv(
                "RANSMAP_CANDIDATE_ACCEPTANCE_CRITERIA"
            ),
            retention_evaluation_definition=os.getenv(
                "RANSMAP_RETENTION_EVALUATION_DEFINITION"
            ),
        )

        config = cls(
            artifacts=artifacts,
            model=model,
            policy=policy,
        )

        config.validate()

        return config

    def validate(self) -> None:
        """
        Validate confirmed structural configuration.

        Unresolved adaptive policy values are intentionally not required here.
        """

        if self.model.selected_checkpoint_identity.strip() == "":
            raise ValueError(
                "selected_checkpoint_identity must not be empty"
            )

        if self.model.device.strip() == "":
            raise ValueError(
                "device must not be empty"
            )

        if self.artifacts.feature_mean is not None:
            if self.artifacts.feature_mean.suffix != ".npy":
                raise ValueError(
                    "feature_mean must reference a .npy file"
                )

        if self.artifacts.feature_std is not None:
            if self.artifacts.feature_std.suffix != ".npy":
                raise ValueError(
                    "feature_std must reference a .npy file"
                )

    def unresolved_policy_settings(self) -> list[str]:
        """
        Return adaptive policy settings that remain unresolved.
        """

        unresolved: list[str] = []

        policy = self.policy

        if policy.drift_formula is None:
            unresolved.append("drift_formula")

        if policy.drift_threshold is None:
            unresolved.append("drift_threshold")

        if policy.drift_monitoring_window is None:
            unresolved.append("drift_monitoring_window")

        if policy.warning_buffer_capacity is None:
            unresolved.append("warning_buffer_capacity")

        if policy.replay_buffer_capacity is None:
            unresolved.append("replay_buffer_capacity")

        if policy.samples_required_before_update is None:
            unresolved.append("samples_required_before_update")

        if policy.replay_ratio is None:
            unresolved.append("replay_ratio")

        if policy.class_balance_strategy is None:
            unresolved.append("class_balance_strategy")

        if policy.learning_rate is None:
            unresolved.append("learning_rate")

        if policy.optimizer is None:
            unresolved.append("optimizer")

        if policy.epochs is None:
            unresolved.append("epochs")

        if policy.batch_size is None:
            unresolved.append("batch_size")

        if policy.candidate_acceptance_criteria is None:
            unresolved.append("candidate_acceptance_criteria")

        if policy.retention_evaluation_definition is None:
            unresolved.append("retention_evaluation_definition")

        return unresolved


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _optional_path_from_env(
    variable_name: str,
) -> Optional[Path]:
    value = os.getenv(variable_name)

    if value is None or value.strip() == "":
        return None

    return Path(value)


def _optional_int_from_env(
    variable_name: str,
) -> Optional[int]:
    value = os.getenv(variable_name)

    if value is None or value.strip() == "":
        return None

    return int(value)


def _optional_float_from_env(
    variable_name: str,
) -> Optional[float]:
    value = os.getenv(variable_name)

    if value is None or value.strip() == "":
        return None

    return float(value)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def build_default_config() -> AdaptiveConfig:
    """
    Build the default configuration without requiring environment variables.
    """

    return AdaptiveConfig(
        artifacts=ArtifactPaths(),
        model=ModelIdentityConfig(
            architecture_loader=None,
            selected_checkpoint_identity=(
                THRESHOLD_TUNED_INTEGRATION_CHECKPOINT
            ),
            model_version=None,
            device="cpu",
        ),
        policy=AdaptivePolicyConfig(),
    )