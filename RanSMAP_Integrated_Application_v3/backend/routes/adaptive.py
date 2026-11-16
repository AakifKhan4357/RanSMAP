from dataclasses import asdict
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.adaptive.adaptive_models import ModelUnavailableError
from backend.adaptive.adaptive_service import (
    AdaptiveServiceError, DuplicateFeedbackError, InvalidFeatureError,
    InvalidLabelError, PredictionNotFoundError, SampleAssociationError,
)
from backend.preprocessing.config import PipelineRejectionError
from backend.services.adaptive_runtime import get_adaptive_service
from backend.services.preprocessing_service import run_preprocessing
from backend.services.quality_service import capture_diagnostics
from backend.services.upload_service import collect_stream_uploads

router = APIRouter()


def _error(code, error_code, message, details=None):
    raise HTTPException(status_code=code, detail={
        "status": "error", "error_code": error_code,
        "message": message, "details": details or {},
    })


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    actual_label: Literal["Benign", "Ransomware"]


@router.post("/adaptive/predict")
async def adaptive_predict(
    ata_read: Optional[UploadFile] = File(None),
    ata_write: Optional[UploadFile] = File(None),
    mem_read: Optional[UploadFile] = File(None),
    mem_write: Optional[UploadFile] = File(None),
    mem_readwrite: Optional[UploadFile] = File(None),
    mem_exec: Optional[UploadFile] = File(None),
    archive: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form("uploaded_sample"),
):
    try:
        files = await collect_stream_uploads(
            ata_read=ata_read, ata_write=ata_write, mem_read=mem_read,
            mem_write=mem_write, mem_readwrite=mem_readwrite,
            mem_exec=mem_exec, archive=archive,
        )
        result = run_preprocessing(
            files=files, sample_id=sample_id or "uploaded_sample",
            label="unlabeled",
        )
    except PipelineRejectionError as exc:
        _error(422, "INVALID_UPLOAD_DATA",
               "The uploaded streams failed preprocessing validation.",
               exc.to_dict())

    try:
        record = get_adaptive_service().predict_one(
            sample_id=sample_id or "uploaded_sample",
            features=result.to_numpy(),
        )
    except ModelUnavailableError:
        _error(503, "ADAPTIVE_UNAVAILABLE",
               "The adaptive model could not be initialized.")
    except InvalidFeatureError as exc:
        _error(422, "INVALID_FEATURES", str(exc))
    except AdaptiveServiceError:
        _error(500, "ADAPTIVE_PREDICTION_FAILED",
               "Adaptive prediction failed.")

    return {
        "sample_id": record.sample_id,
        "prediction_id": record.prediction_id,
        "status": "success",
        "model_type": "Adaptive CNN-LSTM",
        "model_version": record.model_version,
        "num_windows": result.num_windows,
        "num_features": result.num_features,
        "prediction": record.prediction,
        "benign_probability": record.benign_probability,
        "ransomware_probability": record.ransomware_probability,
        "confidence": record.confidence,
        "threshold": record.threshold,
        **capture_diagnostics(result.to_numpy()),
    }


@router.post("/adaptive/feedback")
def adaptive_feedback(request: FeedbackRequest):
    try:
        feedback = get_adaptive_service().reveal_label(
            prediction_id=request.prediction_id,
            sample_id=request.sample_id,
            actual_label=request.actual_label,
        )
    except ModelUnavailableError:
        _error(503, "ADAPTIVE_UNAVAILABLE", "The adaptive model is unavailable.")
    except DuplicateFeedbackError as exc:
        _error(409, "DUPLICATE_FEEDBACK", str(exc))
    except PredictionNotFoundError as exc:
        _error(404, "PREDICTION_NOT_FOUND", str(exc))
    except SampleAssociationError as exc:
        _error(409, "SAMPLE_MISMATCH", str(exc))
    except InvalidLabelError as exc:
        _error(422, "INVALID_LABEL", str(exc))
    except AdaptiveServiceError:
        _error(500, "ADAPTIVE_FEEDBACK_FAILED",
               "Adaptive feedback could not be recorded.")

    response = asdict(feedback)
    response.update({
        "status": "success",
        "feedback_accepted": True,
        "model_updated": False,
        "retention_scope": "in-memory; cleared when the API process restarts",
    })
    return response
