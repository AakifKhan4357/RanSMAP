from fastapi import APIRouter

from backend.services.adaptive_runtime import adaptive_status_payload
from backend.services.baseline_model_loader import load_baseline_model

router = APIRouter()


def _baseline_status():
    try:
        load_baseline_model()
        return {"available": True, "model_type": "CNN-LSTM", "model_version": None}
    except Exception:
        return {"available": False, "model_type": "CNN-LSTM", "model_version": None}


@router.get("/models/status")
def models_status():
    return {"baseline": _baseline_status(), "adaptive": adaptive_status_payload()}


@router.get("/adaptive/status")
def adaptive_status():
    return adaptive_status_payload()
