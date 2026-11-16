from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.preprocessing.config import PipelineRejectionError
from backend.services.prediction_service import predict_features
from backend.services.preprocessing_service import run_preprocessing
from backend.services.quality_service import capture_diagnostics
from backend.services.upload_service import collect_stream_uploads


router = APIRouter()


@router.post("/predict")
async def predict_execution(
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
            ata_read=ata_read,
            ata_write=ata_write,
            mem_read=mem_read,
            mem_write=mem_write,
            mem_readwrite=mem_readwrite,
            mem_exec=mem_exec,
            archive=archive,
        )

        result = run_preprocessing(
            files=files,
            sample_id=sample_id or "uploaded_sample",
            label="unlabeled",
        )
        features = result.to_numpy()
        prediction = predict_features(features)
        diagnostics = capture_diagnostics(features)

        return {
            "sample_id": sample_id or "uploaded_sample",
            "status": "success",
            "model_type": "CNN-LSTM",
            "model_version": None,
            "num_windows": result.num_windows,
            "num_features": result.num_features,
            "prediction": prediction["prediction"],
            "benign_probability": prediction["benign_probability"],
            "ransomware_probability": prediction["ransomware_probability"],
            "confidence": max(
                prediction["benign_probability"],
                prediction["ransomware_probability"],
            ),
            "threshold": prediction["threshold"],
            **diagnostics,
        }

    except PipelineRejectionError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "error_code": "INVALID_UPLOAD_DATA",
                "message": "The uploaded streams failed preprocessing validation.",
                "details": exc.to_dict(),
            },
        )