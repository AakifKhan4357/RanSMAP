from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.adaptive.adaptive_models import ModelUnavailableError
from backend.preprocessing.config import PipelineRejectionError
from backend.services.adaptive_runtime import get_adaptive_service
from backend.services.prediction_service import predict_features
from backend.services.preprocessing_service import run_preprocessing
from backend.services.quality_service import capture_diagnostics
from backend.services.upload_service import collect_stream_uploads

router = APIRouter()


def _error(code, error_code, message, details=None):
    raise HTTPException(status_code=code, detail={
        "status": "error", "error_code": error_code,
        "message": message, "details": details or {},
    })


@router.post("/compare")
async def compare_models(
    ata_read: Optional[UploadFile] = File(None),
    ata_write: Optional[UploadFile] = File(None),
    mem_read: Optional[UploadFile] = File(None),
    mem_write: Optional[UploadFile] = File(None),
    mem_readwrite: Optional[UploadFile] = File(None),
    mem_exec: Optional[UploadFile] = File(None),
    archive: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form("uploaded_sample"),
):
    sample = sample_id or "uploaded_sample"
    try:
        files = await collect_stream_uploads(
            ata_read=ata_read, ata_write=ata_write, mem_read=mem_read,
            mem_write=mem_write, mem_readwrite=mem_readwrite,
            mem_exec=mem_exec, archive=archive,
        )
        result = run_preprocessing(
            files=files, sample_id=sample, label="unlabeled",
        )
    except PipelineRejectionError as exc:
        _error(422, "INVALID_UPLOAD_DATA",
               "The uploaded streams failed preprocessing validation.",
               exc.to_dict())

    baseline_result = predict_features(result.to_numpy())
    baseline = {
        "available": True, "model_type": "CNN-LSTM", "model_version": None,
        "prediction": baseline_result["prediction"],
        "benign_probability": baseline_result["benign_probability"],
        "ransomware_probability": baseline_result["ransomware_probability"],
        "confidence": max(baseline_result["benign_probability"],
                          baseline_result["ransomware_probability"]),
        "threshold": baseline_result["threshold"],
    }
    try:
        adaptive_record = get_adaptive_service().predict_one(
            sample_id=sample, features=result.to_numpy(),
        )
    except ModelUnavailableError:
        _error(503, "ADAPTIVE_UNAVAILABLE",
               "Baseline prediction is ready, but the adaptive model is unavailable.",
               {"sample_id": sample, "baseline": baseline,
                "adaptive": {"available": False}})

    adaptive = {
        "available": True, "prediction_id": adaptive_record.prediction_id,
        "model_type": "Adaptive CNN-LSTM",
        "model_version": adaptive_record.model_version,
        "prediction": adaptive_record.prediction,
        "benign_probability": adaptive_record.benign_probability,
        "ransomware_probability": adaptive_record.ransomware_probability,
        "confidence": adaptive_record.confidence,
        "threshold": adaptive_record.threshold,
    }
    return {
        "status": "success", "sample_id": sample,
        "preprocessing": {
            "num_windows": result.num_windows,
            "num_features": result.num_features,
        },
        **capture_diagnostics(result.to_numpy()),
        "baseline": baseline, "adaptive": adaptive,
        "comparison": {
            "predictions_match": baseline["prediction"] == adaptive["prediction"],
            "same_checkpoint_initialization": False,
            "model_updates_enabled": False,
        },
    }
