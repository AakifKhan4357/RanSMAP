from __future__ import annotations
import copy
import hashlib
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4
import numpy as np
import torch

from .adaptive_config import (
    AdaptiveConfig,
    INPUT_SHAPE,
    RANSOMWARE_THRESHOLD,
    build_default_config,
)
from .adaptive_models import (
    ArtifactCompatibilityError,
    ArtifactLoadError,
    LoadedModel,
    ModelArtifactState,
    ModelRole,
    ModelUnavailableError,
    initialize_model,
    load_pytorch_checkpoint,
    load_saved_checkpoint_metadata,
    resolve_architecture_loader,
    save_checkpoint_atomic,
    train_candidate_model,
)

class AdaptiveServiceError(RuntimeError):
    """Base exception for adaptive-service failures."""

class PredictionNotFoundError(AdaptiveServiceError):
    """Raised when feedback references an unknown prediction."""

class DuplicateFeedbackError(AdaptiveServiceError):
    """Raised when feedback has already been processed."""

class SampleAssociationError(AdaptiveServiceError):
    """Raised when feedback does not match the retained prediction."""

class InvalidLabelError(AdaptiveServiceError):
    """Raised when a feedback label is not supported."""

class InvalidFeatureError(AdaptiveServiceError):
    """Raised when inference features are invalid."""

class CandidateLifecycleError(AdaptiveServiceError):
    """Raised when a candidate operation violates its lifecycle."""

class CandidateValidationError(AdaptiveServiceError):
    """Raised when candidate validation requirements are not satisfied."""

class CandidateNotFoundError(AdaptiveServiceError):
    """Raised when a candidate identifier is unknown."""

class BaselineProtectionError(AdaptiveServiceError):
    """Raised when an operation would overwrite the protected baseline."""

class UnsupportedAdaptiveOperation(AdaptiveServiceError):
    """Raised when an adaptive operation is intentionally not implemented."""

class CandidateStatus(str, Enum):
    TRAINING = "TRAINING"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

@dataclass
class PredictionRecord:
    prediction_id: str
    sample_id: str
    prediction: str
    benign_probability: float
    ransomware_probability: float
    confidence: float
    threshold: float
    model_version: Optional[str]
    features: np.ndarray
    created_at: float = field(
        default_factory=lambda: __import__("time").time()
    )


@dataclass
class RetainedPrediction:
    prediction_id: str
    sample_id: str
    features: np.ndarray
    prediction: str
    ransomware_probability: float
    created_at: float


@dataclass
class VerifiedLabelledSample:
    prediction_id: str
    sample_id: str
    actual_label: str
    predicted_label: str
    prediction_error: float
    features: np.ndarray
    verified_at: float


@dataclass
class FeedbackResult:
    prediction_id: str
    sample_id: str
    actual_label: str
    predicted_label: str
    prediction_error: float
    adaptation_triggered: bool


@dataclass
class AdaptationCandidate:
    candidate_id: str
    model_role: ModelRole
    checkpoint_path: Optional[Path] = None

    training_prediction_ids: List[str] = field(default_factory=list)
    replay_prediction_ids: List[str] = field(default_factory=list)

    status: CandidateStatus = CandidateStatus.TRAINING

    validation_passed: bool = False
    validation_independent: bool = False
    validation_result: Optional[Dict[str, Any]] = None

    model_version: Optional[str] = None
    created_at: float = field(
        default_factory=lambda: __import__("time").time()
    )


@dataclass
class DriftState:
    """
    Drift-monitoring state.

    The actual drift formula and trigger policy remain intentionally
    unconfigured until those project decisions are finalized.
    """

    configured: bool = False
    value: Optional[float] = None
    triggered: bool = False


@dataclass
class AdaptiveStatus:
    available: bool
    ready: bool
    model_type: str
    model_version: Optional[str]
    adaptation_status: str
    pending_feedback_count: int
    total_updates: int


class AdaptiveService:
    """
    Core adaptive backend service.

    This class deliberately remains separate from FastAPI. FastAPI route
    handlers are responsible for translating HTTP requests/responses and
    calling these service methods.

    The service owns:

    - adaptive inference
    - prediction retention
    - feedback verification
    - warning/replay buffers
    - candidate lifecycle
    - candidate validation metadata
    - checkpoint protection
    - active-model lifecycle

    Candidate training infrastructure is present through the approved
    PyTorch training engine, but the public adaptive update path remains
    blocked until the unresolved training and class-balancing policy is
    finalized.

    Candidate validation, approval, promotion, activation, and rollback are
    implemented as controlled model-lifecycle operations. The operational
    drift policy and automatic update trigger remain unresolved.
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        model: Optional[LoadedModel] = None,
    ) -> None:
        self.config = config or build_default_config()

        self._model: Optional[LoadedModel] = model
        self._model_version: Optional[str] = (
            model.model_version if model is not None else None
        )

        self._pending_predictions: Dict[str, PredictionRecord] = {}
        self._feedback_history: Dict[str, FeedbackResult] = {}
        self._verified_samples: Dict[str, VerifiedLabelledSample] = {}

        self._warning_buffer: List[str] = []
        self._replay_buffer: List[str] = []

        self._candidates: Dict[str, AdaptationCandidate] = {}

        # Candidate models remain isolated from the active model.
        #
        # This storage remains private so isolated candidate models cannot
        # become active through this collection directly.
        self._candidate_models: Dict[str, LoadedModel] = {}

        # A rollback target is established only after a candidate has been
        # successfully promoted. It is kept in memory only; restart loses
        # this rollback history.
        self._previous_active_model: Optional[LoadedModel] = None
        self._previous_active_version: Optional[str] = None

        # Counts successful active-model updates only. Candidate training
        # does not increment this counter because training alone does not
        # activate a model.
        self._total_updates = 0

        self._drift_state = DriftState(
            configured=False,
            value=None,
            triggered=False,
        )

    # ------------------------------------------------------------------
    # Model availability / status
    # ------------------------------------------------------------------

    @property
    def model(self) -> Optional[LoadedModel]:
        return self._model

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    def initialize(self) -> LoadedModel:
        """
        Initialize the configured threshold-tuned integration checkpoint.

        This does not fall back to the original Random-State-42 baseline.
        """

        loaded_model = initialize_model(config=self.config)

        self._model = loaded_model
        self._model_version = loaded_model.model_version

        return loaded_model

    def status(self) -> AdaptiveStatus:
        return AdaptiveStatus(
            available=self._model is not None,
            ready=self._model is not None,
            model_type="adaptive",
            model_version=self._model_version,
            adaptation_status=(
                "ready"
                if self._model is not None
                else "unavailable"
            ),
            pending_feedback_count=len(self._pending_predictions),
            total_updates=self._total_updates,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_one(
        self,
        sample_id: str,
        features: np.ndarray,
    ) -> PredictionRecord:
        """
        Predict one unlabelled execution.

        Actual labels are deliberately not accepted by this method.

        Prediction sequence:

        1. Validate sample_id.
        2. Validate feature matrix.
        3. Run model inference.
        4. Convert logits to probabilities.
        5. Apply the agreed ransomware threshold of 0.30.
        6. Generate prediction_id.
        7. Retain the prediction and feature matrix for feedback.
        """

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise InvalidFeatureError(
                "sample_id must be a non-empty string."
            )

        if self._model is None:
            raise ModelUnavailableError(
                "Adaptive model is unavailable. "
                "Initialize the configured integration checkpoint first."
            )

        validated_features = self._validate_features(features)

        logits = self._model.predict_logits(validated_features)

        # The approved LoadedModel.predict_logits() returns a detached
        # CPU torch.Tensor. as_tensor() remains intentionally compatible
        # with tensor-like/NumPy implementations without changing the
        # service contract.
        logits = torch.as_tensor(
            logits,
            dtype=torch.float32,
        )

        if tuple(logits.shape) != (1, 2):
            raise AdaptiveServiceError(
                f"Expected logits shape (1, 2), got {tuple(logits.shape)}."
            )

        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)

        if tuple(probabilities.shape) != (1, 2):
            raise AdaptiveServiceError(
                "Probability tensor does not have expected shape (1, 2)."
            )

        if not torch.isfinite(probabilities).all().item():
            raise AdaptiveServiceError(
                "Model returned non-finite class probabilities."
            )

        benign_probability = float(
            probabilities[0, 0].item()
        )
        ransomware_probability = float(
            probabilities[0, 1].item()
        )

        predicted_label = (
            "Ransomware"
            if ransomware_probability >= RANSOMWARE_THRESHOLD
            else "Benign"
        )

        confidence = (
            ransomware_probability
            if predicted_label == "Ransomware"
            else benign_probability
        )

        prediction_id = self._new_prediction_id()

        record = PredictionRecord(
            prediction_id=prediction_id,
            sample_id=sample_id,
            prediction=predicted_label,
            benign_probability=benign_probability,
            ransomware_probability=ransomware_probability,
            confidence=confidence,
            threshold=RANSOMWARE_THRESHOLD,
            model_version=self._model_version,
            features=validated_features.copy(),
        )

        self._pending_predictions[prediction_id] = record

        return record

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def reveal_label(
        self,
        prediction_id: str,
        sample_id: str,
        actual_label: str,
    ) -> FeedbackResult:
        """
        Reveal the actual label after prediction.

        Feedback is tied to the exact prediction_id and verified against
        sample_id before the labelled feature matrix is retained.
        """

        if prediction_id in self._feedback_history:
            raise DuplicateFeedbackError(
                f"Feedback for prediction_id '{prediction_id}' "
                "has already been processed."
            )

        record = self._pending_predictions.get(prediction_id)

        if record is None:
            raise PredictionNotFoundError(
                f"No pending prediction exists for prediction_id "
                f"'{prediction_id}'."
            )

        if record.sample_id != sample_id:
            raise SampleAssociationError(
                f"sample_id '{sample_id}' does not match the prediction's "
                f"sample_id '{record.sample_id}'."
            )

        normalized_label = self._normalize_label(actual_label)

        prediction_error = self._calculate_prediction_error(
            prediction=record.prediction,
            actual_label=normalized_label,
        )

        verified = VerifiedLabelledSample(
            prediction_id=record.prediction_id,
            sample_id=record.sample_id,
            actual_label=normalized_label,
            predicted_label=record.prediction,
            prediction_error=prediction_error,
            features=record.features.copy(),
            verified_at=__import__("time").time(),
        )

        self._verified_samples[prediction_id] = verified

        self._update_buffers(prediction_id)

        adaptation_triggered = self._check_adaptation_trigger()

        result = FeedbackResult(
            prediction_id=prediction_id,
            sample_id=sample_id,
            actual_label=normalized_label,
            predicted_label=record.prediction,
            prediction_error=prediction_error,
            adaptation_triggered=adaptation_triggered,
        )

        self._feedback_history[prediction_id] = result

        del self._pending_predictions[prediction_id]

        return result

    # ------------------------------------------------------------------
    # Candidate lifecycle
    # ------------------------------------------------------------------

    def create_candidate(
        self,
        training_prediction_ids: Sequence[str],
        replay_prediction_ids: Optional[Sequence[str]] = None,
        model_role: ModelRole = ModelRole.CANDIDATE,
    ) -> AdaptationCandidate:
        """
        Create an adaptive candidate from already verified samples.

        Candidate creation does not perform model training.
        """

        training_ids = list(training_prediction_ids)
        replay_ids = list(replay_prediction_ids or [])

        self._validate_verified_prediction_ids(training_ids)
        self._validate_verified_prediction_ids(replay_ids)

        candidate_id = self._new_candidate_id()

        candidate = AdaptationCandidate(
            candidate_id=candidate_id,
            model_role=model_role,
            training_prediction_ids=training_ids,
            replay_prediction_ids=replay_ids,
            status=CandidateStatus.TRAINING,
        )

        self._candidates[candidate_id] = candidate

        return candidate

    def record_candidate_validation(
        self,
        candidate_id: str,
        *,
        passed: bool,
        independent: bool,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> AdaptationCandidate:
        """
        Record independent validation of a candidate checkpoint.

        A candidate can only become VALIDATED after independent validation
        explicitly passes.
        """

        candidate = self._get_candidate(candidate_id)

        if candidate.status in {
            CandidateStatus.APPROVED,
            CandidateStatus.REJECTED,
            CandidateStatus.FAILED,
        }:
            raise CandidateLifecycleError(
                f"Candidate {candidate_id} cannot be validated while "
                f"in status {candidate.status.value}."
            )

        if not independent:
            raise CandidateValidationError(
                "Candidate validation must be independent."
            )

        # Verify the checkpoint before changing any validation-related
        # candidate state. A failed attempt must not leave partial metadata.
        if candidate.checkpoint_path is None:
            raise CandidateValidationError(
                "A candidate checkpoint must be saved before validation "
                "can be recorded."
            )

        checkpoint_path = (
            Path(candidate.checkpoint_path).expanduser().resolve()
        )
        if not checkpoint_path.is_file():
            raise CandidateValidationError(
                f"Candidate checkpoint does not exist: {checkpoint_path}"
            )

        candidate.validation_independent = True
        candidate.validation_passed = bool(passed)
        candidate.validation_result = (
            dict(validation_result)
            if validation_result is not None
            else None
        )

        if passed:
            candidate.status = CandidateStatus.VALIDATED
        else:
            candidate.status = CandidateStatus.REJECTED

        return candidate

    def save_candidate_checkpoint(
        self,
        candidate_id: str,
        model: LoadedModel,
        checkpoint_path: os.PathLike[str] | str,
        *,
        model_version: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save the candidate checkpoint.

        Lifecycle rule:

        TRAINING -> checkpoint can be saved
        VALIDATED -> checkpoint replacement is forbidden
        APPROVED/REJECTED/FAILED -> checkpoint saving is forbidden

        This prevents replacing the checkpoint after independent validation.
        """

        candidate = self._get_candidate(candidate_id)

        # IMPORTANT:
        # A VALIDATED candidate is now immutable with respect to its
        # checkpoint. The exact checkpoint that was independently validated
        # must be the checkpoint subsequently loaded and activated.
        if candidate.status in {
            CandidateStatus.VALIDATED,
            CandidateStatus.APPROVED,
            CandidateStatus.REJECTED,
            CandidateStatus.FAILED,
        }:
            raise CandidateLifecycleError(
                f"Candidate {candidate_id} cannot save a checkpoint "
                f"while in status {candidate.status.value}."
            )

        target = Path(checkpoint_path).expanduser().resolve()

        if candidate.checkpoint_path is not None:
            existing = candidate.checkpoint_path.resolve()

            if existing != target:
                raise ArtifactCompatibilityError(
                    f"Candidate {candidate_id} already has checkpoint path "
                    f"'{existing}'. A candidate checkpoint cannot be "
                    f"redirected to '{target}'."
                )

        baseline_path = (
            self.config.artifacts.baseline_random_state_42_checkpoint
        )

        if baseline_path is not None:
            baseline_resolved = (
                Path(baseline_path).expanduser().resolve()
            )

            if target == baseline_resolved:
                raise BaselineProtectionError(
                    "Candidate checkpoint cannot overwrite the protected "
                    "original Random-State-42 baseline checkpoint."
                )

        approved_integration_path = self._approved_checkpoint_path()

        if approved_integration_path is not None:
            if target == approved_integration_path.resolve():
                raise BaselineProtectionError(
                    "Candidate checkpoint cannot overwrite the configured "
                    "active integration checkpoint."
                )

        resolved_model_version = (
            model_version
            if model_version is not None
            else model.model_version
        )
        if (
            resolved_model_version is None
            or not isinstance(resolved_model_version, str)
            or not resolved_model_version.strip()
        ):
            raise CandidateValidationError(
                "Candidate checkpoint requires a non-empty model version."
            )

        if model.normalization is None:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint requires normalization artifacts."
            )

        architecture_loader = self.config.model.architecture_loader
        if not architecture_loader:
            raise ModelUnavailableError(
                "No architecture factory is configured for candidate "
                "checkpoint persistence."
            )

        normalization_identity = self._normalization_identity(
            model.normalization
        )

        metadata: Dict[str, Any] = dict(extra_metadata or {})
        protected_metadata = {
            "candidate_id",
            "architecture_loader",
            "feature_mean_identity",
            "feature_std_identity",
            "validated",
            "training_prediction_ids",
            "replay_prediction_ids",
        }
        conflicting_metadata = protected_metadata.intersection(
            metadata.keys()
        )
        if conflicting_metadata:
            raise ArtifactCompatibilityError(
                "extra_metadata contains protected candidate metadata keys: "
                + ", ".join(sorted(conflicting_metadata))
            )

        metadata.update({
            "candidate_id": candidate.candidate_id,
            "architecture_loader": architecture_loader,
            "feature_mean_identity": normalization_identity["feature_mean"],
            "feature_std_identity": normalization_identity["feature_std"],
            "validated": False,
            "training_prediction_ids": list(
                candidate.training_prediction_ids
            ),
            "replay_prediction_ids": list(
                candidate.replay_prediction_ids
            ),
        })

        save_checkpoint_atomic(
            loaded_model=model,
            checkpoint_path=target,
            checkpoint_identity=f"candidate:{candidate.candidate_id}",
            model_version=resolved_model_version,
            model_role=ModelRole.CANDIDATE,
            extra_metadata=metadata,
        )

        candidate.checkpoint_path = target
        candidate.model_version = resolved_model_version

        # Candidate remains TRAINING until independent validation explicitly
        # promotes it to VALIDATED.
        candidate.status = CandidateStatus.TRAINING

        return target

    def approve_candidate(
        self,
        candidate_id: str,
    ) -> AdaptationCandidate:
        """
        Load, verify, and promote an independently validated candidate.

        This is the single public approval workflow. The candidate checkpoint
        is loaded from its exact saved path and independently checked against
        its persisted metadata, configured architecture, normalization
        identity, candidate identity, and version before activation.
        """

        candidate = self._get_candidate(candidate_id)

        self._validate_candidate_for_approval(candidate)

        loaded_candidate = self._load_candidate_checkpoint(candidate)

        self._validate_loaded_candidate(candidate, loaded_candidate)

        return self._activate_verified_candidate(
            candidate_id,
            loaded_candidate,
        )

    def approve_loaded_candidate(
        self,
        candidate_id: str,
        loaded_candidate: LoadedModel,
    ) -> AdaptationCandidate:
        """
        Activate a pre-loaded candidate only after complete verification.

        This method is retained for compatibility with the existing service
        contract, but it is not an approval bypass: the supplied model must
        match the exact persisted candidate checkpoint and all Stage 3
        metadata/compatibility requirements before activation.
        """

        candidate = self._get_candidate(candidate_id)

        self._validate_candidate_for_approval(candidate)

        self._validate_loaded_candidate(candidate, loaded_candidate)

        return self._activate_verified_candidate(
            candidate_id,
            loaded_candidate,
        )

    def _validate_candidate_for_approval(
        self,
        candidate: AdaptationCandidate,
    ) -> None:
        if candidate.status != CandidateStatus.VALIDATED:
            raise CandidateLifecycleError(
                f"Candidate {candidate.candidate_id} must be VALIDATED before "
                f"activation; current status is {candidate.status.value}."
            )

        if candidate.model_role != ModelRole.CANDIDATE:
            raise CandidateValidationError(
                "Only candidates with model_role=CANDIDATE can be activated."
            )

        if candidate.checkpoint_path is None:
            raise CandidateValidationError(
                "Validated candidate has no checkpoint path."
            )

        if not candidate.validation_passed:
            raise CandidateValidationError(
                "Candidate validation did not pass."
            )

        if not candidate.validation_independent:
            raise CandidateValidationError(
                "Candidate validation was not independent."
            )

        checkpoint_path = candidate.checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise CandidateValidationError(
                f"Candidate checkpoint does not exist: {checkpoint_path}"
            )

        if not candidate.model_version or not isinstance(
            candidate.model_version, str
        ):
            raise CandidateValidationError(
                "Validated candidate must have a non-empty model version."
            )

    def _load_candidate_checkpoint(
        self,
        candidate: AdaptationCandidate,
    ) -> LoadedModel:
        """Load the exact candidate checkpoint without any fallback path."""

        checkpoint_path = candidate.checkpoint_path
        if checkpoint_path is None:
            raise CandidateValidationError(
                "Candidate has no checkpoint path."
            )

        metadata = load_saved_checkpoint_metadata(checkpoint_path)

        expected_loader = self.config.model.architecture_loader
        if not expected_loader:
            raise ModelUnavailableError(
                "No architecture factory is configured for candidate "
                "activation."
            )

        configured_normalization = self._load_configured_normalization()
        normalization_identity = self._normalization_identity(
            configured_normalization
        )

        self._validate_candidate_metadata(
            candidate,
            metadata,
            expected_loader=expected_loader,
            normalization_identity=normalization_identity,
        )

        factory = resolve_architecture_loader(expected_loader)

        loaded_raw = load_pytorch_checkpoint(
            checkpoint_path,
            model_factory=factory,
            device=self.config.model.device,
        )

        loaded_raw.normalization = configured_normalization
        loaded_raw.artifact_state = ModelArtifactState(
            checkpoint_identity=str(metadata["checkpoint_identity"]),
            model_version=str(metadata["model_version"]),
            role=ModelRole.CANDIDATE,
        )

        return loaded_raw

    def _validate_candidate_metadata(
        self,
        candidate: AdaptationCandidate,
        metadata: Dict[str, Any],
        *,
        expected_loader: str,
        normalization_identity: Dict[str, str],
    ) -> None:
        required = (
            "checkpoint_identity",
            "candidate_id",
            "model_role",
            "model_version",
            "architecture_loader",
            "feature_mean_identity",
            "feature_std_identity",
            "input_shape",
            "num_classes",
            "class_mapping",
        )

        missing = [name for name in required if name not in metadata]
        if missing:
            raise CandidateValidationError(
                "Candidate checkpoint metadata is incomplete: "
                + ", ".join(missing)
            )

        if metadata["candidate_id"] != candidate.candidate_id:
            raise CandidateValidationError(
                "Candidate checkpoint metadata does not belong to the "
                f"candidate '{candidate.candidate_id}'."
            )

        if metadata["model_role"] != ModelRole.CANDIDATE.value:
            raise CandidateValidationError(
                "Candidate checkpoint metadata does not identify a "
                "candidate model."
            )

        if metadata["model_version"] != candidate.model_version:
            raise CandidateValidationError(
                "Candidate checkpoint model version does not match the "
                "validated candidate version."
            )

        if metadata["architecture_loader"] != expected_loader:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint architecture loader does not match "
                "the configured RanSMAP architecture loader."
            )

        if metadata["feature_mean_identity"] != normalization_identity["feature_mean"]:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint feature_mean identity does not match "
                "the configured normalization artifact."
            )

        if metadata["feature_std_identity"] != normalization_identity["feature_std"]:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint feature_std identity does not match "
                "the configured normalization artifact."
            )

        if metadata["input_shape"] != list(INPUT_SHAPE):
            raise ArtifactCompatibilityError(
                "Candidate checkpoint input shape is incompatible with "
                f"{INPUT_SHAPE}."
            )

        if metadata["num_classes"] != 2:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint must define exactly two classes."
            )

        if metadata["class_mapping"] != {
            "0": "Benign",
            "1": "Ransomware",
        }:
            raise ArtifactCompatibilityError(
                "Candidate checkpoint class mapping is incompatible with "
                "the RanSMAP binary class contract."
            )

        expected_identity = f"candidate:{candidate.candidate_id}"
        if metadata["checkpoint_identity"] != expected_identity:
            raise CandidateValidationError(
                "Candidate checkpoint identity does not match the "
                "candidate identifier."
            )

    def _validate_loaded_candidate(
        self,
        candidate: AdaptationCandidate,
        loaded_candidate: LoadedModel,
    ) -> None:
        """Verify an externally supplied loaded candidate against its file."""

        if not isinstance(loaded_candidate, LoadedModel):
            raise CandidateValidationError(
                "loaded_candidate must be a LoadedModel instance."
            )

        if candidate.checkpoint_path is None:
            raise CandidateValidationError(
                "Candidate has no checkpoint path."
            )

        metadata = load_saved_checkpoint_metadata(candidate.checkpoint_path)
        normalization = loaded_candidate.normalization
        if normalization is None:
            raise ArtifactCompatibilityError(
                "Loaded candidate has no normalization artifacts."
            )

        expected_loader = self.config.model.architecture_loader
        if not expected_loader:
            raise ModelUnavailableError(
                "No architecture factory is configured for candidate "
                "activation."
            )

        expected_norm = self._load_configured_normalization()
        if self._normalization_identity(normalization) != self._normalization_identity(expected_norm):
            raise ArtifactCompatibilityError(
                "Loaded candidate normalization does not match the configured "
                "candidate normalization artifacts."
            )

        self._validate_candidate_metadata(
            candidate,
            metadata,
            expected_loader=expected_loader,
            normalization_identity=self._normalization_identity(expected_norm),
        )

        loaded_state = loaded_candidate.artifact_state
        if loaded_state is None:
            raise CandidateValidationError(
                "Loaded candidate has no artifact identity metadata."
            )

        if loaded_state.checkpoint_identity != metadata["checkpoint_identity"]:
            raise CandidateValidationError(
                "Loaded candidate checkpoint identity does not match the "
                "persisted candidate checkpoint."
            )

        if loaded_state.model_version != candidate.model_version:
            raise CandidateValidationError(
                "Loaded candidate model version does not match the "
                "validated candidate version."
            )

        if loaded_state.role != ModelRole.CANDIDATE:
            raise CandidateValidationError(
                "Loaded candidate is not marked as a candidate model."
            )

        checkpoint_state_identity = self._checkpoint_state_identity(
            candidate.checkpoint_path
        )
        loaded_state_identity = self._model_state_identity(
            loaded_candidate.model.state_dict()
        )
        if loaded_state_identity != checkpoint_state_identity:
            raise CandidateValidationError(
                "Loaded candidate parameters do not match the exact "
                "persisted candidate checkpoint."
            )

        # Ensure the supplied model is structurally compatible by checking its
        # forward contract before activation. This deliberately uses a zero
        # feature tensor only for shape validation and does not infer a label.
        probe = np.zeros(INPUT_SHAPE, dtype=np.float64)
        logits = loaded_candidate.predict_logits(probe)
        if tuple(logits.shape) != (1, 2):
            raise ArtifactCompatibilityError(
                "Loaded candidate must produce logits with shape (1, 2)."
            )
        if not torch.isfinite(torch.as_tensor(logits)).all().item():
            raise ArtifactCompatibilityError(
                "Loaded candidate produced non-finite logits during "
                "compatibility validation."
            )

    def _activate_verified_candidate(
        self,
        candidate_id: str,
        loaded_candidate: LoadedModel,
    ) -> AdaptationCandidate:
        """Perform the final atomic in-memory promotion step."""

        candidate = self._get_candidate(candidate_id)

        previous_model = self._model
        previous_version = self._model_version

        if previous_model is None:
            raise ModelUnavailableError(
                "Cannot promote a candidate because no active model exists "
                "to establish a rollback target."
            )

        try:
            activated_state = loaded_candidate.artifact_state
            if activated_state is None:
                raise CandidateValidationError(
                    "Verified candidate has no artifact identity metadata."
                )

            loaded_candidate.artifact_state = ModelArtifactState(
                checkpoint_identity=activated_state.checkpoint_identity,
                model_version=activated_state.model_version,
                role=ModelRole.ACTIVE,
            )

            self._model = loaded_candidate
            self._model_version = loaded_candidate.model_version
            candidate.status = CandidateStatus.APPROVED
            self._total_updates += 1

            # The rollback reference is changed only after activation and the
            # update counter succeed. Failed promotion therefore leaves the
            # previous active model and rollback state untouched.
            self._previous_active_model = previous_model
            self._previous_active_version = previous_version

            return candidate

        except Exception:
            self._model = previous_model
            self._model_version = previous_version
            raise

    def reject_candidate(
        self,
        candidate_id: str,
        reason: Optional[str] = None,
    ) -> AdaptationCandidate:
        candidate = self._get_candidate(candidate_id)

        if candidate.status == CandidateStatus.APPROVED:
            raise CandidateLifecycleError(
                f"Approved candidate {candidate_id} cannot be rejected."
            )

        if candidate.status == CandidateStatus.REJECTED:
            raise CandidateLifecycleError(
                f"Candidate {candidate_id} is already rejected."
            )

        if candidate.status == CandidateStatus.FAILED:
            raise CandidateLifecycleError(
                f"Failed candidate {candidate_id} cannot be rejected."
            )

        candidate.status = CandidateStatus.REJECTED

        if reason is not None:
            candidate.validation_result = {
                **(candidate.validation_result or {}),
                "rejection_reason": reason,
            }

        return candidate

    # ------------------------------------------------------------------
    # Active checkpoint lifecycle
    # ------------------------------------------------------------------

    def save_active_checkpoint(
        self,
        checkpoint_path: os.PathLike[str] | str,
        *,
        model_version: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save the currently active adaptive model.

        The original Random-State-42 baseline checkpoint is protected.
        """

        if self._model is None:
            raise ModelUnavailableError(
                "Cannot save an active checkpoint because no adaptive "
                "model is loaded."
            )

        target = Path(checkpoint_path).expanduser().resolve()

        baseline_path = (
            self.config.artifacts.baseline_random_state_42_checkpoint
        )

        if baseline_path is not None:
            baseline_resolved = (
                Path(baseline_path).expanduser().resolve()
            )

            if target == baseline_resolved:
                raise BaselineProtectionError(
                    "Active adaptive checkpoint cannot overwrite the "
                    "protected original Random-State-42 baseline."
                )

        metadata: Dict[str, Any] = {
            "model_version": (
                model_version
                if model_version is not None
                else self._model_version
            ),
            "active": True,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        save_checkpoint_atomic(
            loaded_model=self._model,
            checkpoint_path=target,
            model_version=(
                model_version
                if model_version is not None
                else self._model_version
            ),
            model_role=ModelRole.ACTIVE,
            extra_metadata=metadata,
        )

        return target

    def load_checkpoint(
        self,
        checkpoint_path: os.PathLike[str] | str,
    ) -> LoadedModel:
        """
        Load the configured adaptive integration checkpoint.

        The supplied path is checked against the configured integration
        checkpoint. A different checkpoint is never silently substituted.

        The configured integration checkpoint remains the only initialization
        target here; there is no baseline fallback.
        """

        requested_path = (
            Path(checkpoint_path).expanduser().resolve()
        )

        configured_path = (
            self.config.artifacts.integration_checkpoint
        )

        if configured_path is None:
            raise ModelUnavailableError(
                "No configured adaptive integration checkpoint exists."
            )

        configured_resolved = (
            Path(configured_path).expanduser().resolve()
        )

        if requested_path != configured_resolved:
            raise ArtifactCompatibilityError(
                "Requested checkpoint does not match the configured "
                "adaptive integration checkpoint. Refusing to silently "
                "load a different checkpoint."
            )

        loaded_model = initialize_model(config=self.config)

        self._model = loaded_model
        self._model_version = loaded_model.model_version

        return loaded_model

    def rollback(self) -> LoadedModel:
        """Restore the previous active model kept from the last promotion."""

        target_model = self._previous_active_model
        target_version = self._previous_active_version

        if target_model is None:
            raise UnsupportedAdaptiveOperation(
                "Rollback is unavailable because no previous active model "
                "has been established by a successful candidate promotion."
            )

        if target_model.artifact_state is None:
            raise CandidateValidationError(
                "Rollback target has no artifact identity metadata."
            )

        if target_model.model_version != target_version:
            raise ArtifactCompatibilityError(
                "Rollback target version does not match its saved rollback "
                "version."
            )

        current_model = self._model
        current_version = self._model_version

        if current_model is None:
            raise ModelUnavailableError(
                "Cannot rollback because no active model is loaded."
            )

        try:
            self._model = target_model
            self._model_version = target_version

            # Swap the references so rollback is reversible without creating
            # a new persistent history structure. total_updates deliberately
            # does not change.
            self._previous_active_model = current_model
            self._previous_active_version = current_version

            return target_model

        except Exception:
            self._model = current_model
            self._model_version = current_version
            raise

    # ------------------------------------------------------------------
    # Adaptive update entry point
    # ------------------------------------------------------------------

    def update_model(
        self,
        training_prediction_ids: Sequence[str],
        replay_prediction_ids: Optional[Sequence[str]] = None,
        *,
        reason: Optional[str] = None,
    ) -> None:
        """
        Train one isolated adaptive candidate from verified feedback.

        This method validates the inputs and required training policy for an
        adaptive update, but the current implementation deliberately stops
        before candidate training because the project has not finalized the
        class-balancing policy or its service implementation. No model
        weights are modified by this method while that policy is unresolved.
        """

        if self._model is None:
            raise ModelUnavailableError(
                "Adaptive model is unavailable. Initialize the configured "
                "integration checkpoint before training a candidate."
            )

        training_ids = self._unique_prediction_ids(
            training_prediction_ids
        )

        replay_ids = self._unique_prediction_ids(
            replay_prediction_ids or []
        )

        self._validate_verified_prediction_ids(training_ids)
        self._validate_verified_prediction_ids(replay_ids)

        required_settings = (
            self._required_training_policy_settings()
        )

        unresolved = [
            name
            for name in required_settings
            if getattr(self.config.policy, name) is None
        ]

        if unresolved:
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because required "
                "policy settings are unresolved: "
                f"{', '.join(unresolved)}."
            )

        policy = self.config.policy

        if len(training_ids) < policy.samples_required_before_update:
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because the number "
                f"of verified training samples ({len(training_ids)}) is "
                "below the configured minimum "
                f"({policy.samples_required_before_update})."
            )

        if (
            not math.isfinite(policy.replay_ratio)
            or not 0.0 <= policy.replay_ratio <= 1.0
        ):
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because "
                "replay_ratio must be a finite value between 0.0 and 1.0."
            )

        if (
            not isinstance(policy.optimizer, str)
            or not policy.optimizer.strip()
        ):
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because optimizer "
                "is not a valid configured string."
            )

        # The project has not yet defined the semantics or implementation
        # contract of class_balance_strategy. A non-empty string alone is
        # not enough to safely choose a balancing algorithm, so the adaptive
        # update path blocks training until that policy is explicitly defined.
        if (
            not isinstance(
                policy.class_balance_strategy,
                str,
            )
            or not policy.class_balance_strategy.strip()
        ):
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because "
                "class_balance_strategy is not configured."
            )

        raise UnsupportedAdaptiveOperation(
            "Adaptive model update cannot proceed because the configured "
            "class_balance_strategy has no finalized service "
            "implementation. The project team must define the balancing "
            "semantics before training can be enabled."
        )

        # The training path below is intentionally retained as the concrete
        # Integration point for the approved training engine once the
        # unresolved class-balancing policy has been finalized.
        #
        # It becomes reachable once the unresolved class-balancing policy
        # has been finalized and its implementation contract is supplied.
        #
        # selected_replay_ids = self._select_replay_ids(
        #     training_ids,
        #     replay_ids,
        #     replay_ratio=policy.replay_ratio,
        # )
        #
        # selected_ids = training_ids + selected_replay_ids
        #
        # features, labels = self._build_training_arrays(selected_ids)
        #
        # candidate_id = self._new_candidate_id()
        #
        # candidate = AdaptationCandidate(
        #     candidate_id=candidate_id,
        #     model_role=ModelRole.CANDIDATE,
        #     training_prediction_ids=list(training_ids),
        #     replay_prediction_ids=list(selected_replay_ids),
        #     status=CandidateStatus.TRAINING,
        # )
        #
        # self._candidates[candidate_id] = candidate
        #
        # try:
        #     trained_candidate, training_report = train_candidate_model(
        #         self._model,
        #         features,
        #         labels,
        #         learning_rate=policy.learning_rate,
        #         optimizer_name=policy.optimizer,
        #         epochs=policy.epochs,
        #         batch_size=policy.batch_size,
        #     )
        # except Exception as exc:
        #     candidate.status = CandidateStatus.FAILED
        #     candidate.validation_result = {
        #         "training_error": str(exc),
        #         "reason": reason,
        #     }
        #     raise
        #
        # self._candidate_models[candidate_id] = trained_candidate
        #
        # candidate.validation_result = {
        #     "training_report": training_report,
        #     "reason": reason,
        #     "sample_count": len(selected_ids),
        # }
        #
        # Candidate remains TRAINING. No active-model update counter
        # changes because no promotion or activation occurs in this update
        # path while training remains blocked by unresolved policy.

    # ------------------------------------------------------------------
    # Verified sample / buffer access
    # ------------------------------------------------------------------

    def get_verified_sample(
        self,
        prediction_id: str,
    ) -> VerifiedLabelledSample:
        try:
            return self._verified_samples[prediction_id]
        except KeyError as exc:
            raise PredictionNotFoundError(
                f"No verified sample exists for prediction_id "
                f"'{prediction_id}'."
            ) from exc

    def get_warning_buffer(self) -> List[str]:
        return list(self._warning_buffer)

    def get_replay_buffer(self) -> List[str]:
        return list(self._replay_buffer)

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_features(
        features: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            raise InvalidFeatureError(
                "features must be provided as a NumPy array."
            )

        if np.iscomplexobj(features):
            raise InvalidFeatureError(
                "Complex-valued features are not supported."
            )

        if not np.issubdtype(features.dtype, np.number):
            raise InvalidFeatureError(
                "features must contain numeric values."
            )

        if tuple(features.shape) != INPUT_SHAPE:
            raise InvalidFeatureError(
                f"Expected feature shape {INPUT_SHAPE}, "
                f"got {tuple(features.shape)}."
            )

        if not np.isfinite(features).all():
            raise InvalidFeatureError(
                "features contain non-finite values."
            )

        return np.asarray(
            features,
            dtype=np.float64,
        ).copy()

    @staticmethod
    def _normalize_label(label: str) -> str:
        if not isinstance(label, str):
            raise InvalidLabelError(
                "actual_label must be a string."
            )

        normalized = label.strip()

        if normalized not in {
            "Benign",
            "Ransomware",
        }:
            raise InvalidLabelError(
                "actual_label must be exactly "
                "'Benign' or 'Ransomware'."
            )

        return normalized

    @staticmethod
    def _calculate_prediction_error(
        prediction: str,
        actual_label: str,
    ) -> float:
        return (
            0.0
            if prediction == actual_label
            else 1.0
        )

    def _validate_verified_prediction_ids(
        self,
        prediction_ids: Iterable[str],
    ) -> None:
        missing = [
            prediction_id
            for prediction_id in prediction_ids
            if prediction_id not in self._verified_samples
        ]

        if missing:
            raise PredictionNotFoundError(
                "The following prediction_ids have not been verified: "
                + ", ".join(missing)
            )

    def _update_buffers(
        self,
        prediction_id: str,
    ) -> None:
        """
        Retain verified predictions for replay.

        Warning-buffer entry is deliberately disabled until the project's
        drift policy defines when a verified prediction qualifies as a warning.
        Buffer capacities and finalized update-trigger policy are not invented.
        """

        # Verified feedback is retained for replay immediately. Warning-buffer
        # membership is withheld until the drift policy defines its entry rule.
        if prediction_id not in self._replay_buffer:
            self._replay_buffer.append(prediction_id)

    def _check_adaptation_trigger(self) -> bool:
        """
        Return whether the finalized adaptive policy currently requests an
        update.

        No drift formula or trigger threshold has been finalized, so this
        remains false until that policy is explicitly configured.
        """

        return False

    @staticmethod
    def _required_training_policy_settings() -> Tuple[str, ...]:
        """
        Return policy fields required to perform candidate training.

        Drift detection, candidate acceptance, and retention definitions
        are deliberately excluded because they belong to later lifecycle
        stages rather than the actual PyTorch training operation.
        """

        return (
            "samples_required_before_update",
            "replay_ratio",
            "class_balance_strategy",
            "learning_rate",
            "optimizer",
            "epochs",
            "batch_size",
        )

    @staticmethod
    def _unique_prediction_ids(
        prediction_ids: Sequence[str],
    ) -> List[str]:
        """
        Deduplicate prediction IDs while preserving caller order.
        """

        unique: List[str] = []
        seen: set[str] = set()

        for prediction_id in prediction_ids:
            if (
                not isinstance(prediction_id, str)
                or not prediction_id.strip()
            ):
                raise PredictionNotFoundError(
                    "prediction_id values must be non-empty strings."
                )

            if prediction_id not in seen:
                seen.add(prediction_id)
                unique.append(prediction_id)

        return unique

    def _select_replay_ids(
        self,
        training_ids: Sequence[str],
        replay_ids: Sequence[str],
        *,
        replay_ratio: float,
    ) -> List[str]:
        """
        Select replay IDs from explicitly supplied IDs or the retained
        replay buffer.

        This helper does not perform class balancing or replacement
        sampling. Its interpretation of replay_ratio is deliberately kept
        narrow: it determines the requested replay count relative to the
        new-training sample count.

        The helper is currently unreachable while the class-balance policy
        remains unresolved.
        """

        if replay_ratio <= 0.0:
            return []

        available = [
            prediction_id
            for prediction_id in self._replay_buffer
            if prediction_id not in training_ids
        ]

        if replay_ids:
            requested = [
                prediction_id
                for prediction_id in replay_ids
                if prediction_id not in training_ids
            ]
            available = requested

        if not available:
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot proceed because the "
                "configured replay_ratio requires replay samples, but "
                "no eligible replay samples were supplied or retained."
            )

        if replay_ratio < 1.0:
            desired = math.ceil(
                (
                    replay_ratio
                    / (1.0 - replay_ratio)
                )
                * len(training_ids)
            )
        else:
            desired = len(available)

        if desired > len(available):
            raise UnsupportedAdaptiveOperation(
                "Adaptive model update cannot satisfy the configured "
                f"replay_ratio: requires {desired} replay samples but "
                f"only {len(available)} eligible replay samples are "
                "available."
            )

        return list(available[:desired])

    def _build_training_arrays(
        self,
        prediction_ids: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build raw feature/label arrays from verified feedback records.
        """

        samples = [
            self._verified_samples[prediction_id]
            for prediction_id in prediction_ids
        ]

        features = np.stack(
            [
                sample.features
                for sample in samples
            ],
            axis=0,
        )

        labels = np.asarray(
            [
                (
                    0
                    if sample.actual_label == "Benign"
                    else 1
                )
                for sample in samples
            ],
            dtype=np.int64,
        )

        expected_shape = (
            len(samples),
        ) + INPUT_SHAPE

        if tuple(features.shape) != expected_shape:
            raise InvalidFeatureError(
                "Verified feedback features do not match the required "
                f"training shape {expected_shape}; got "
                f"{tuple(features.shape)}."
            )

        return features, labels

    def _load_configured_normalization(self):
        """Load the configured mean/std artifacts through the Stage 1 contract."""

        if self._model is not None and self._model.normalization is not None:
            return self._model.normalization

        # Candidate approval requires an active normalization contract.
        # initialize_model() is intentionally not called here because it would
        # reload the integration checkpoint, which is forbidden for candidate
        # loading.
        raise ModelUnavailableError(
            "Active normalization artifacts are unavailable for candidate "
            "compatibility validation."
        )

    @staticmethod
    def _normalization_identity(normalization) -> Dict[str, str]:
        return {
            "feature_mean": hashlib.sha256(
                np.ascontiguousarray(normalization.feature_mean).tobytes()
            ).hexdigest(),
            "feature_std": hashlib.sha256(
                np.ascontiguousarray(normalization.feature_std).tobytes()
            ).hexdigest(),
        }

    @staticmethod
    def _model_state_identity(state_dict: Dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for key in sorted(state_dict):
            tensor = state_dict[key].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _checkpoint_state_identity(self, checkpoint_path: Path) -> str:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise ArtifactLoadError(
                "The installed PyTorch version does not support "
                "weights_only=True checkpoint loading mode."
            ) from exc
        except Exception as exc:
            raise CandidateValidationError(
                f"Unable to safely inspect candidate checkpoint: {checkpoint_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise CandidateValidationError(
                "Candidate checkpoint payload is not a mapping."
            )

        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise CandidateValidationError(
                "Candidate checkpoint does not contain model_state_dict."
            )

        if not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise CandidateValidationError(
                "Candidate checkpoint model_state_dict is invalid."
            )

        return self._model_state_identity(state_dict)

    def _get_candidate(
        self,
        candidate_id: str,
    ) -> AdaptationCandidate:
        try:
            return self._candidates[candidate_id]
        except KeyError as exc:
            raise CandidateNotFoundError(
                f"Unknown candidate_id '{candidate_id}'."
            ) from exc

    def _approved_checkpoint_path(self) -> Optional[Path]:
        candidate_path = (
            self.config.artifacts.integration_checkpoint
        )

        if candidate_path is None:
            return None

        return Path(candidate_path).expanduser().resolve()

    @staticmethod
    def _new_prediction_id() -> str:
        return f"prediction_{uuid4().hex}"

    @staticmethod
    def _new_candidate_id() -> str:
        return f"candidate_{uuid4().hex}"