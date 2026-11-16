from dataclasses import asdict
from pathlib import Path
from threading import Lock

from backend.adaptive.adaptive_config import AdaptiveConfig, AdaptivePolicyConfig, ArtifactPaths, ModelIdentityConfig
from backend.adaptive.adaptive_models import ModelUnavailableError
from backend.adaptive.adaptive_service import AdaptiveService

_service = None
_init_error = None
_lock = Lock()


def get_adaptive_service():
    global _service, _init_error
    if _service is not None:
        return _service
    with _lock:
        if _service is not None:
            return _service
        if _init_error is not None:
            raise ModelUnavailableError("Adaptive model initialization failed.") from _init_error
        model_dir = Path(__file__).resolve().parents[1] / "models"
        config = AdaptiveConfig(
            artifacts=ArtifactPaths(
                integration_checkpoint=model_dir / "threshold_model.pth.zip",
                feature_mean=model_dir / "feature_mean.npy",
                feature_std=model_dir / "feature_std.npy",
                baseline_random_state_42_checkpoint=(
                    model_dir / "ransmap_cnn_lstm_original_td30_tw0p1.pkl"
                ),
            ),
            model=ModelIdentityConfig(
                architecture_loader="backend.services.model_loader:RanSMAP_CNN_LSTM",
                selected_checkpoint_identity="threshold_tuned_cnn_lstm_integration_checkpoint",
                model_version=None,
                device="cpu",
            ),
            policy=AdaptivePolicyConfig(),
        )
        service = AdaptiveService(config=config)
        try:
            service.initialize()
        except Exception as exc:
            _init_error = exc
            raise ModelUnavailableError("Adaptive model initialization failed.") from exc
        _service = service
        return service


def adaptive_status_payload():
    try:
        service = get_adaptive_service()
    except ModelUnavailableError:
        return {
            "available": False,
            "ready": False,
            "model_type": "Adaptive CNN-LSTM",
            "model_version": None,
            "adaptation_status": "unavailable",
            "pending_feedback_count": 0,
            "total_updates": 0,
            "reason": "The integration checkpoint or normalization artifacts could not be loaded.",
            "feedback_retention": "unavailable",
            "model_updates_enabled": False,
        }
    status = asdict(service.status())
    status.update({
        "model_type": "Adaptive CNN-LSTM",
        "feedback_retention": "in-memory; cleared when the API process restarts",
        "model_updates_enabled": False,
        "update_policy_status": "unconfigured",
    })
    return status
