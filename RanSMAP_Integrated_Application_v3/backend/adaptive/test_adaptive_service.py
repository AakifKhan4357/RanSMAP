"""Focused tests for the PS-RanSMAP adaptive service.
The tests use small deterministic PyTorch fixtures and synthetic normalization
artifacts. They do not require the RanSMAP dataset, the real CNN-LSTM
checkpoint, live adaptive training, or operational drift detection. Stage 3
model promotion and rollback lifecycle behavior is tested with isolated
fixtures.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from .adaptive_config import (
    AdaptiveConfig,
    AdaptivePolicyConfig,
    ArtifactPaths,
    ModelIdentityConfig,
)
from .adaptive_models import (
    ArtifactCompatibilityError,
    ArtifactLoadError,
    LoadedModel,
    ModelArtifactState,
    ModelRole,
    NormalizationArtifacts,
    load_saved_checkpoint_metadata,
    save_checkpoint_atomic,
)
from .adaptive_service import (
    AdaptiveService,
    BaselineProtectionError,
    CandidateLifecycleError,
    CandidateStatus,
    CandidateValidationError,
    DuplicateFeedbackError,
    InvalidFeatureError,
    InvalidLabelError,
    ModelUnavailableError,
    PredictionNotFoundError,
    SampleAssociationError,
    UnsupportedAdaptiveOperation,
)


INPUT_SHAPE = (300, 23)
NUM_CLASSES = 2
THRESHOLD = 0.30


class FixedLogitModel(nn.Module):
    """Small deterministic PyTorch model used only by these tests."""

    def __init__(self, logits: tuple[float, float]) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.bias.unsqueeze(0)


class TensorReturningLoadedModel(LoadedModel):
    """LoadedModel test double matching the service's current logits contract.

    The approved LoadedModel.predict_logits() implementation returns a
    detached CPU torch.Tensor. This test double preserves that logits contract
    while keeping the service tests deterministic and isolated from the real
    RanSMAP model architecture.
    """

    def __init__(
        self,
        model: nn.Module,
        normalization: NormalizationArtifacts,
        *,
        model_version: str | None = "test-model",
    ) -> None:
        super().__init__(
            model=model,
            normalization=normalization,
            device=torch.device("cpu"),
            artifact_state=ModelArtifactState(
                checkpoint_identity="test_checkpoint",
                model_version=model_version,
                role=ModelRole.ACTIVE,
            ),
        )

    @property
    def model_version(self) -> str | None:
        return self.artifact_state.model_version if self.artifact_state else None

    def predict_logits(self, features: np.ndarray) -> torch.Tensor:
        normalized = self.normalization.normalize(features)
        tensor = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)

        return logits


def build_test_model() -> nn.Module:
    """Architecture factory used when testing actual checkpoint initialization."""
    return FixedLogitModel((0.0, 1.0))


@pytest.fixture
def normalization() -> NormalizationArtifacts:
    mean = np.zeros(INPUT_SHAPE[1], dtype=np.float64)
    std = np.ones(INPUT_SHAPE[1], dtype=np.float64)
    return NormalizationArtifacts(feature_mean=mean, feature_std=std)


@pytest.fixture
def features() -> np.ndarray:
    return np.zeros(INPUT_SHAPE, dtype=np.float64)


@pytest.fixture
def service_factory():
    def factory(
        model: LoadedModel | None = None,
        *,
        integration_checkpoint: Path | None = None,
        feature_mean: Path | None = None,
        feature_std: Path | None = None,
        baseline_checkpoint: Path | None = None,
        architecture_loader: str | None = None,
        model_version: str | None = "test-model",
    ) -> AdaptiveService:
        config = AdaptiveConfig(
            artifacts=ArtifactPaths(
                integration_checkpoint=integration_checkpoint,
                feature_mean=feature_mean,
                feature_std=feature_std,
                baseline_random_state_42_checkpoint=baseline_checkpoint,
            ),
            model=ModelIdentityConfig(
                architecture_loader=architecture_loader,
                model_version=model_version,
                device="cpu",
            ),
            policy=AdaptivePolicyConfig(),
        )
        return AdaptiveService(config=config, model=model)

    return factory


def fixed_service(
    normalization: NormalizationArtifacts,
    logits: tuple[float, float] = (0.0, 0.0),
    *,
    model_version: str | None = "test-model",
) -> AdaptiveService:
    model = TensorReturningLoadedModel(
        FixedLogitModel(logits),
        normalization,
        model_version=model_version,
    )
    config = AdaptiveConfig(
        artifacts=ArtifactPaths(),
        model=ModelIdentityConfig(
            architecture_loader=f"{__package__}.test_adaptive_service:build_test_model",
            model_version=model_version,
            device="cpu",
        ),
        policy=AdaptivePolicyConfig(),
    )
    return AdaptiveService(config=config, model=model)


def create_checkpoint(
    path: Path,
    normalization: NormalizationArtifacts,
    *,
    model_version: str = "test-model",
    logits: tuple[float, float] = (0.0, 1.0),
) -> None:
    loaded = LoadedModel(
        model=FixedLogitModel(logits),
        normalization=normalization,
        device=torch.device("cpu"),
        artifact_state=ModelArtifactState(
            checkpoint_identity="test_checkpoint",
            model_version=model_version,
            role=ModelRole.ACTIVE,
        ),
    )

    save_checkpoint_atomic(
        loaded_model=loaded,
        checkpoint_path=path,
        model_version=model_version,
        model_role=ModelRole.ACTIVE,
    )


def rewrite_checkpoint_metadata(path: Path, **updates: object) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = dict(payload["metadata"])
    metadata.update(updates)
    payload["metadata"] = metadata
    torch.save(payload, path)


def write_normalization_files(
    tmp_path: Path,
) -> tuple[Path, Path]:
    mean_path = tmp_path / "feature_mean.npy"
    std_path = tmp_path / "feature_std.npy"

    np.save(mean_path, np.zeros(INPUT_SHAPE[1], dtype=np.float64))
    np.save(std_path, np.ones(INPUT_SHAPE[1], dtype=np.float64))

    return mean_path, std_path


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


def test_initialization_success(
    tmp_path: Path,
    normalization: NormalizationArtifacts,
    service_factory,
) -> None:
    checkpoint_path = tmp_path / "integration.pth"
    create_checkpoint(checkpoint_path, normalization)

    mean_path, std_path = write_normalization_files(tmp_path)

    service = service_factory(
        integration_checkpoint=checkpoint_path,
        feature_mean=mean_path,
        feature_std=std_path,
        architecture_loader=f"{__package__}.test_adaptive_service:build_test_model",
    )

    loaded = service.initialize()

    assert isinstance(loaded, LoadedModel)
    assert service.model is loaded
    assert service.model_version == "test-model"
    assert service.status().available is True
    assert service.status().ready is True


def test_initialization_reports_missing_integration_artifact(
    tmp_path: Path,
    service_factory,
) -> None:
    mean_path, std_path = write_normalization_files(tmp_path)

    service = service_factory(
        integration_checkpoint=tmp_path / "missing.pth",
        feature_mean=mean_path,
        feature_std=std_path,
        architecture_loader=f"{__package__}.test_adaptive_service:build_test_model",
    )

    with pytest.raises(ArtifactLoadError):
        service.initialize()


# ---------------------------------------------------------------------------
# 2. Prediction (predict_one)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("logits", "expected_label"),
    [
        ((1.0, 0.0), "Benign"),
        ((0.0, math.log(0.30 / 0.70) + 1e-6), "Ransomware"),
        ((0.0, 1.0), "Ransomware"),
    ],
)
def test_predict_one_applies_probability_mapping_and_threshold(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    logits: tuple[float, float],
    expected_label: str,
) -> None:
    service = fixed_service(normalization, logits=logits)

    result = service.predict_one("execution-001", features)

    assert result.sample_id == "execution-001"
    assert result.prediction == expected_label
    assert result.threshold == pytest.approx(THRESHOLD)
    assert result.prediction_id.startswith("prediction_")
    assert result.features.shape == INPUT_SHAPE
    assert np.array_equal(result.features, features)

    assert result.benign_probability + result.ransomware_probability == pytest.approx(1.0)

    if expected_label == "Ransomware":
        assert result.ransomware_probability >= THRESHOLD
        assert result.confidence == pytest.approx(result.ransomware_probability)
    else:
        assert result.ransomware_probability < THRESHOLD
        assert result.confidence == pytest.approx(result.benign_probability)


def test_predict_one_retains_prediction_for_feedback(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization, logits=(0.0, 1.0))

    result = service.predict_one("execution-002", features)

    assert result.prediction_id in service._pending_predictions
    retained = service._pending_predictions[result.prediction_id]
    assert retained.sample_id == "execution-002"
    assert np.array_equal(retained.features, features)


@pytest.mark.parametrize(
    "bad_features",
    [
        np.zeros((299, 23), dtype=np.float64),
        np.zeros((300, 22), dtype=np.float64),
        np.full(INPUT_SHAPE, np.nan, dtype=np.float64),
        np.full(INPUT_SHAPE, np.inf, dtype=np.float64),
        np.zeros(INPUT_SHAPE, dtype=np.complex128),
        np.zeros(INPUT_SHAPE, dtype=object),
    ],
)
def test_predict_one_rejects_invalid_feature_contract(
    normalization: NormalizationArtifacts,
    bad_features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    with pytest.raises(InvalidFeatureError):
        service.predict_one("execution-invalid", bad_features)


def test_predict_one_rejects_empty_sample_id(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    with pytest.raises(InvalidFeatureError):
        service.predict_one("   ", features)


def test_predict_one_requires_an_available_model(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = AdaptiveService()

    with pytest.raises(ModelUnavailableError):
        service.predict_one("execution-no-model", features)


# ---------------------------------------------------------------------------
# 3. Feedback (reveal_label)
# ---------------------------------------------------------------------------


def test_reveal_label_records_feedback_and_removes_pending_prediction(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization, logits=(0.0, 1.0))

    prediction = service.predict_one("execution-feedback", features)

    result = service.reveal_label(
        prediction.prediction_id,
        "execution-feedback",
        "Ransomware",
    )

    assert result.prediction_id == prediction.prediction_id
    assert result.sample_id == "execution-feedback"
    assert result.actual_label == "Ransomware"
    assert result.predicted_label == "Ransomware"
    assert result.prediction_error == pytest.approx(0.0)
    assert result.adaptation_triggered is False

    assert prediction.prediction_id not in service._pending_predictions

    verified = service.get_verified_sample(prediction.prediction_id)
    assert verified.sample_id == "execution-feedback"
    assert verified.actual_label == "Ransomware"
    assert verified.predicted_label == "Ransomware"
    assert verified.prediction_error == pytest.approx(0.0)
    assert np.array_equal(verified.features, features)


def test_reveal_label_records_prediction_error_for_mismatch(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization, logits=(0.0, 1.0))

    prediction = service.predict_one("execution-error", features)

    result = service.reveal_label(
        prediction.prediction_id,
        "execution-error",
        "Benign",
    )

    assert result.predicted_label == "Ransomware"
    assert result.actual_label == "Benign"
    assert result.prediction_error == pytest.approx(1.0)


@pytest.mark.parametrize("label", ["Benign", "Ransomware"])
def test_reveal_label_accepts_only_supported_labels(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    label: str,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-label", features)

    result = service.reveal_label(
        prediction.prediction_id,
        "execution-label",
        label,
    )

    assert result.actual_label == label


def test_reveal_label_rejects_unknown_prediction_id(
    normalization: NormalizationArtifacts,
) -> None:
    service = fixed_service(normalization)

    with pytest.raises(PredictionNotFoundError):
        service.reveal_label(
            "prediction_missing",
            "execution-missing",
            "Benign",
        )


def test_reveal_label_rejects_wrong_sample_association(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-correct", features)

    with pytest.raises(SampleAssociationError):
        service.reveal_label(
            prediction.prediction_id,
            "execution-wrong",
            "Benign",
        )


def test_reveal_label_rejects_invalid_label(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-invalid-label", features)

    with pytest.raises(InvalidLabelError):
        service.reveal_label(
            prediction.prediction_id,
            "execution-invalid-label",
            "benign",
        )

    assert prediction.prediction_id in service._pending_predictions


def test_reveal_label_rejects_duplicate_feedback(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-duplicate", features)

    service.reveal_label(
        prediction.prediction_id,
        "execution-duplicate",
        "Benign",
    )

    with pytest.raises(DuplicateFeedbackError):
        service.reveal_label(
            prediction.prediction_id,
            "execution-duplicate",
            "Benign",
        )


# ---------------------------------------------------------------------------
# 4. Warning/replay buffer behaviour
# ---------------------------------------------------------------------------


def test_feedback_updates_warning_and_replay_buffers(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-buffer", features)

    result = service.reveal_label(
        prediction.prediction_id,
        "execution-buffer",
        "Benign",
    )

    assert result.adaptation_triggered is False
    assert prediction.prediction_id not in service.get_warning_buffer()
    assert prediction.prediction_id in service.get_replay_buffer()


def test_current_adaptation_trigger_remains_false(
    normalization: NormalizationArtifacts,
) -> None:
    service = fixed_service(normalization)

    assert service._check_adaptation_trigger() is False


# ---------------------------------------------------------------------------
# 5. Candidate creation
# ---------------------------------------------------------------------------


def test_candidate_creation_uses_verified_prediction_ids(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    first = service.predict_one("execution-candidate-1", features)
    second = service.predict_one("execution-candidate-2", features)

    service.reveal_label(first.prediction_id, first.sample_id, "Benign")
    service.reveal_label(second.prediction_id, second.sample_id, "Ransomware")

    candidate = service.create_candidate(
        training_prediction_ids=[first.prediction_id],
        replay_prediction_ids=[second.prediction_id],
    )

    assert candidate.candidate_id.startswith("candidate_")
    assert candidate.status == CandidateStatus.TRAINING
    assert candidate.training_prediction_ids == [first.prediction_id]
    assert candidate.replay_prediction_ids == [second.prediction_id]
    assert candidate.validation_passed is False
    assert candidate.validation_independent is False


def test_candidate_creation_rejects_unverified_prediction_ids(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    pending = service.predict_one("execution-pending", features)

    with pytest.raises(PredictionNotFoundError):
        service.create_candidate([pending.prediction_id])


# ---------------------------------------------------------------------------
# 6. Candidate validation
# ---------------------------------------------------------------------------


def test_candidate_validation_requires_independent_validation(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-validation", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")

    candidate = service.create_candidate([prediction.prediction_id])

    with pytest.raises(CandidateValidationError):
        service.record_candidate_validation(
            candidate.candidate_id,
            passed=True,
            independent=False,
        )

    assert candidate.status == CandidateStatus.TRAINING


def test_candidate_cannot_be_validated_before_checkpoint_is_saved(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-no-checkpoint", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")

    candidate = service.create_candidate([prediction.prediction_id])

    with pytest.raises(CandidateValidationError):
        service.record_candidate_validation(
            candidate.candidate_id,
            passed=True,
            independent=True,
            validation_result={"validation_loss": 0.1},
        )

    assert candidate.status == CandidateStatus.TRAINING


def test_candidate_validation_promotes_saved_candidate_to_validated(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-validated", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")

    candidate = service.create_candidate([prediction.prediction_id])

    checkpoint_path = tmp_path / "candidate.pth"
    service.save_candidate_checkpoint(
        candidate.candidate_id,
        fixed_service(normalization).model,
        checkpoint_path,
        model_version="candidate-v1",
        extra_metadata={"validation_scope": "unit-test"},
    )

    validated = service.record_candidate_validation(
        candidate.candidate_id,
        passed=True,
        independent=True,
        validation_result={"validation_loss": 0.1},
    )

    assert validated.status == CandidateStatus.VALIDATED
    assert validated.validation_passed is True
    assert validated.validation_independent is True
    assert validated.validation_result == {
        "validation_loss": 0.1,
    }


# ---------------------------------------------------------------------------
# 7. Candidate checkpoint saving
# ---------------------------------------------------------------------------


def test_candidate_checkpoint_saving_persists_metadata(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-save", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])

    checkpoint_path = tmp_path / "candidate-save.pth"
    saved_path = service.save_candidate_checkpoint(
        candidate.candidate_id,
        service.model,
        checkpoint_path,
        model_version="candidate-v2",
        extra_metadata={"test_marker": "candidate-save"},
    )

    assert saved_path == checkpoint_path.resolve()
    assert saved_path.exists()
    assert candidate.checkpoint_path == saved_path
    assert candidate.model_version == "candidate-v2"
    assert candidate.status == CandidateStatus.TRAINING

    metadata = load_saved_checkpoint_metadata(saved_path)

    assert metadata["candidate_id"] == candidate.candidate_id
    assert metadata["model_role"] == ModelRole.CANDIDATE.value
    assert metadata["model_version"] == "candidate-v2"
    assert metadata["validated"] is False
    assert metadata["test_marker"] == "candidate-save"


def test_validated_candidate_checkpoint_cannot_be_replaced(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-immutable", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])

    checkpoint_path = tmp_path / "candidate-immutable.pth"
    service.save_candidate_checkpoint(
        candidate.candidate_id,
        service.model,
        checkpoint_path,
    )
    service.record_candidate_validation(
        candidate.candidate_id,
        passed=True,
        independent=True,
    )

    with pytest.raises(CandidateLifecycleError):
        service.save_candidate_checkpoint(
            candidate.candidate_id,
            service.model,
            checkpoint_path,
        )


# ---------------------------------------------------------------------------
# 8. Stage 3 candidate promotion / rollback
# ---------------------------------------------------------------------------


def prepare_validated_candidate(
    service: AdaptiveService,
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
    *,
    version: str = "candidate-v2",
):
    prediction = service.predict_one("execution-stage3", features)
    service.reveal_label(
        prediction.prediction_id,
        prediction.sample_id,
        "Benign",
    )
    candidate = service.create_candidate([prediction.prediction_id])
    checkpoint_path = tmp_path / f"{candidate.candidate_id}.pth"
    service.save_candidate_checkpoint(
        candidate.candidate_id,
        service.model,
        checkpoint_path,
        model_version=version,
    )
    service.record_candidate_validation(
        candidate.candidate_id,
        passed=True,
        independent=True,
        validation_result={"validation_loss": 0.1},
    )
    return candidate, checkpoint_path


def test_approve_candidate_requires_validated_candidate(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
):
    service = fixed_service(normalization)
    prediction = service.predict_one("execution-not-validated", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])

    with pytest.raises(CandidateLifecycleError):
        service.approve_candidate(candidate.candidate_id)

    assert service.model_version == "test-model"
    assert service.status().total_updates == 0


def test_approve_candidate_successfully_promotes_exact_checkpoint(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, logits=(0.0, 0.0), model_version="active-v1")
    candidate, checkpoint_path = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    old_model = service.model

    activated = service.approve_candidate(candidate.candidate_id)

    assert activated.status == CandidateStatus.APPROVED
    assert service.model is not old_model
    assert service.model_version == "candidate-v2"
    assert service.model.artifact_state is not None
    assert service.model.artifact_state.role == ModelRole.ACTIVE
    assert service.model.artifact_state.checkpoint_identity == f"candidate:{candidate.candidate_id}"
    assert service.status().total_updates == 1
    assert checkpoint_path.exists()


def test_candidate_without_explicit_model_version_can_be_validated_and_promoted(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    prediction = service.predict_one("execution-version-default", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])
    checkpoint_path = tmp_path / "candidate-default-version.pth"

    service.save_candidate_checkpoint(
        candidate.candidate_id,
        service.model,
        checkpoint_path,
    )

    assert candidate.model_version == "active-v1"
    metadata = load_saved_checkpoint_metadata(checkpoint_path)
    assert metadata["model_version"] == "active-v1"

    service.record_candidate_validation(
        candidate.candidate_id,
        passed=True,
        independent=True,
    )
    activated = service.approve_candidate(candidate.candidate_id)

    assert activated.status == CandidateStatus.APPROVED
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 1


def test_failed_validation_cannot_be_promoted(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    prediction = service.predict_one("execution-failed-validation", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])
    checkpoint_path = tmp_path / "failed-validation.pth"
    service.save_candidate_checkpoint(
        candidate.candidate_id, service.model, checkpoint_path, model_version="candidate-v2"
    )
    service.record_candidate_validation(
        candidate.candidate_id, passed=False, independent=True
    )

    with pytest.raises(CandidateLifecycleError):
        service.approve_candidate(candidate.candidate_id)

    assert candidate.status == CandidateStatus.REJECTED
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_candidate_checkpoint_missing_after_validation_cannot_promote(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, checkpoint_path = prepare_validated_candidate(
        service, normalization, features, tmp_path
    )
    checkpoint_path.unlink()

    with pytest.raises(CandidateValidationError):
        service.approve_candidate(candidate.candidate_id)

    assert candidate.status == CandidateStatus.VALIDATED
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_corrupt_candidate_checkpoint_cannot_promote(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, checkpoint_path = prepare_validated_candidate(
        service, normalization, features, tmp_path
    )
    checkpoint_path.write_bytes(b"not-a-pytorch-checkpoint")

    with pytest.raises(ArtifactLoadError):
        service.approve_candidate(candidate.candidate_id)

    assert candidate.status == CandidateStatus.VALIDATED
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_approve_loaded_candidate_rejects_mismatched_parameters(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )

    mismatched = TensorReturningLoadedModel(
        FixedLogitModel((9.0, -9.0)),
        normalization,
        model_version="candidate-v2",
    )
    mismatched.artifact_state = ModelArtifactState(
        checkpoint_identity=f"candidate:{candidate.candidate_id}",
        model_version="candidate-v2",
        role=ModelRole.CANDIDATE,
    )

    with pytest.raises(CandidateValidationError, match="parameters do not match"):
        service.approve_loaded_candidate(candidate.candidate_id, mismatched)

    assert candidate.status == CandidateStatus.VALIDATED
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_candidate_rejection_leaves_active_model_unchanged(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    old_model = service.model
    candidate, _ = prepare_validated_candidate(service, normalization, features, tmp_path)

    service.reject_candidate(candidate.candidate_id, reason="validation-policy")

    assert candidate.status == CandidateStatus.REJECTED
    assert service.model is old_model
    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_successful_promotion_establishes_rollback_reference_once(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    old_model = service.model
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )

    service.approve_candidate(candidate.candidate_id)

    assert service.status().total_updates == 1
    assert service.model_version == "candidate-v2"
    assert service._previous_active_model is old_model
    assert service._previous_active_version == "active-v1"


def test_failed_promotion_preserves_active_version_and_rollback_state(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    first, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    service.approve_candidate(first.candidate_id)

    previous_model = service._previous_active_model
    previous_version = service._previous_active_version
    active_model = service.model
    active_version = service.model_version
    updates = service.status().total_updates

    second_prediction = service.predict_one("execution-stage3-second", features)
    service.reveal_label(second_prediction.prediction_id, second_prediction.sample_id, "Benign")
    second = service.create_candidate([second_prediction.prediction_id])
    second_path = tmp_path / "second.pth"
    service.save_candidate_checkpoint(
        second.candidate_id, service.model, second_path, model_version="candidate-v3"
    )
    service.record_candidate_validation(second.candidate_id, passed=True, independent=True)
    second_path.write_bytes(b"corrupt-after-validation")

    with pytest.raises(ArtifactLoadError):
        service.approve_candidate(second.candidate_id)

    assert service.model is active_model
    assert service.model_version == active_version
    assert service.status().total_updates == updates
    assert service._previous_active_model is previous_model
    assert service._previous_active_version == previous_version


def test_candidate_approval_rejects_mismatched_checkpoint_metadata(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, checkpoint_path = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    rewrite_checkpoint_metadata(
        checkpoint_path,
        candidate_id="candidate-not-this-candidate",
    )

    with pytest.raises(CandidateValidationError):
        service.approve_candidate(candidate.candidate_id)

    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_candidate_approval_rejects_mismatched_normalization_artifacts(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )

    mismatched_mean = np.ones(INPUT_SHAPE[1], dtype=np.float64)
    service.model.normalization = NormalizationArtifacts(
        feature_mean=mismatched_mean,
        feature_std=np.ones(INPUT_SHAPE[1], dtype=np.float64),
    )

    with pytest.raises(ArtifactCompatibilityError):
        service.approve_candidate(candidate.candidate_id)

    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


def test_approve_loaded_candidate_accepts_correctly_loaded_candidate(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    loaded_candidate = service._load_candidate_checkpoint(candidate)

    activated = service.approve_loaded_candidate(
        candidate.candidate_id,
        loaded_candidate,
    )

    assert activated.status == CandidateStatus.APPROVED
    assert service.model is loaded_candidate
    assert service.model_version == "candidate-v2"
    assert service.status().total_updates == 1


def test_successful_rollback_restores_previous_active_model_without_increment(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    original_model = service.model
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    service.approve_candidate(candidate.candidate_id)
    promoted_model = service.model

    updates_before = service.status().total_updates
    restored = service.rollback()

    assert restored is original_model
    assert service.model is original_model
    assert service.model_version == "active-v1"
    assert service.status().total_updates == updates_before
    assert service._previous_active_model is promoted_model
    assert service._previous_active_version == "candidate-v2"


def test_failed_rollback_preserves_state_for_invalid_rollback_target(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, _ = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    service.approve_candidate(candidate.candidate_id)

    active_model = service.model
    active_version = service.model_version
    updates = service.status().total_updates
    rollback_model = service._previous_active_model
    rollback_version = service._previous_active_version

    assert rollback_model is not None
    assert rollback_version == "active-v1"
    rollback_model.artifact_state = None

    with pytest.raises(CandidateValidationError):
        service.rollback()

    assert service.model is active_model
    assert service.model_version == active_version
    assert service.status().total_updates == updates
    assert service._previous_active_model is rollback_model
    assert service._previous_active_version == rollback_version


def test_repeated_approval_and_invalid_lifecycle_are_rejected(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, _ = prepare_validated_candidate(service, normalization, features, tmp_path)
    service.approve_candidate(candidate.candidate_id)

    with pytest.raises(CandidateLifecycleError):
        service.approve_candidate(candidate.candidate_id)

    with pytest.raises(CandidateLifecycleError):
        service.reject_candidate(candidate.candidate_id)

    assert service.status().total_updates == 1


def test_candidate_approval_does_not_fallback_to_integration_or_baseline(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
):
    service = fixed_service(normalization, model_version="active-v1")
    candidate, candidate_path = prepare_validated_candidate(
        service, normalization, features, tmp_path, version="candidate-v2"
    )
    candidate_path.unlink()

    # The configured integration/baseline artifacts are irrelevant here:
    # approval must use the candidate's exact checkpoint path and must not
    # substitute another model when that checkpoint is unavailable.
    with pytest.raises(CandidateValidationError):
        service.approve_candidate(candidate.candidate_id)

    assert service.model_version == "active-v1"
    assert service.status().total_updates == 0


# ---------------------------------------------------------------------------
# 9. Baseline checkpoint protection
# ---------------------------------------------------------------------------


def test_candidate_checkpoint_cannot_overwrite_baseline(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "original_random_state_42.pth"
    baseline_path.write_bytes(b"protected-baseline")

    service = fixed_service(normalization)
    service.config = AdaptiveConfig(
        artifacts=ArtifactPaths(
            baseline_random_state_42_checkpoint=baseline_path,
        ),
        model=ModelIdentityConfig(device="cpu"),
        policy=AdaptivePolicyConfig(),
    )

    prediction = service.predict_one("execution-baseline", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")
    candidate = service.create_candidate([prediction.prediction_id])

    with pytest.raises(BaselineProtectionError):
        service.save_candidate_checkpoint(
            candidate.candidate_id,
            service.model,
            baseline_path,
        )

    assert baseline_path.read_bytes() == b"protected-baseline"


def test_active_checkpoint_cannot_overwrite_baseline(
    normalization: NormalizationArtifacts,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline-active.pth"
    baseline_path.write_bytes(b"protected-baseline")

    service = fixed_service(normalization)
    service.config = AdaptiveConfig(
        artifacts=ArtifactPaths(
            baseline_random_state_42_checkpoint=baseline_path,
        ),
        model=ModelIdentityConfig(device="cpu"),
        policy=AdaptivePolicyConfig(),
    )

    with pytest.raises(BaselineProtectionError):
        service.save_active_checkpoint(baseline_path)

    assert baseline_path.read_bytes() == b"protected-baseline"


# ---------------------------------------------------------------------------
# 10. Checkpoint loading restrictions
# ---------------------------------------------------------------------------


def test_load_checkpoint_rejects_checkpoint_that_is_not_configured_integration_path(
    tmp_path: Path,
    service_factory,
) -> None:
    integration_path = tmp_path / "integration.pth"
    other_path = tmp_path / "other.pth"

    service = service_factory(
        integration_checkpoint=integration_path,
        feature_mean=tmp_path / "feature_mean.npy",
        feature_std=tmp_path / "feature_std.npy",
        architecture_loader=f"{__package__}.test_adaptive_service:build_test_model",
    )

    with pytest.raises(ArtifactCompatibilityError):
        service.load_checkpoint(other_path)


def test_load_checkpoint_does_not_fallback_to_baseline(
    tmp_path: Path,
    service_factory,
) -> None:
    baseline_path = tmp_path / "baseline.pth"

    service = service_factory(
        integration_checkpoint=None,
        baseline_checkpoint=baseline_path,
    )

    with pytest.raises(ModelUnavailableError):
        service.load_checkpoint(baseline_path)


def test_load_checkpoint_initializes_only_configured_integration_artifact(
    tmp_path: Path,
    normalization: NormalizationArtifacts,
    service_factory,
) -> None:
    integration_path = tmp_path / "integration.pth"
    baseline_path = tmp_path / "baseline.pth"
    mean_path, std_path = write_normalization_files(tmp_path)

    create_checkpoint(integration_path, normalization)
    create_checkpoint(baseline_path, normalization)

    service = service_factory(
        integration_checkpoint=integration_path,
        feature_mean=mean_path,
        feature_std=std_path,
        baseline_checkpoint=baseline_path,
        architecture_loader=f"{__package__}.test_adaptive_service:build_test_model",
    )

    loaded = service.load_checkpoint(integration_path)

    assert isinstance(loaded, LoadedModel)
    assert service.model is loaded

# ---------------------------------------------------------------------------
# 11. Unsupported adaptive operations
# ---------------------------------------------------------------------------

def test_update_model_is_unsupported_when_policy_is_unresolved(
    normalization: NormalizationArtifacts,
    features: np.ndarray,
) -> None:
    service = fixed_service(normalization)

    prediction = service.predict_one("execution-update", features)
    service.reveal_label(prediction.prediction_id, prediction.sample_id, "Benign")

    with pytest.raises(UnsupportedAdaptiveOperation):
        service.update_model([prediction.prediction_id])


def test_rollback_requires_a_previous_active_model(
    normalization: NormalizationArtifacts,
) -> None:
    service = fixed_service(normalization)

    active_model = service.model
    active_version = service.model_version
    updates = service.status().total_updates

    with pytest.raises(UnsupportedAdaptiveOperation):
        service.rollback()

    assert service.model is active_model
    assert service.model_version == active_version
    assert service.status().total_updates == updates
    assert service._previous_active_model is None
    assert service._previous_active_version is None
