from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.preprocessing_service import run_preprocessing
from backend.services.upload_service import collect_stream_uploads
from backend.preprocessing.config import PipelineRejectionError

router = APIRouter()


@router.post("/preprocess")
async def preprocess_execution(
    ata_read: Optional[UploadFile] = File(None),
    ata_write: Optional[UploadFile] = File(None),
    mem_read: Optional[UploadFile] = File(None),
    mem_write: Optional[UploadFile] = File(None),
    mem_readwrite: Optional[UploadFile] = File(None),
    mem_exec: Optional[UploadFile] = File(None),
    archive: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form("test_sample"),
    label: str = Form("benign"),
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
            sample_id=sample_id or "test_sample",
            label=label,
        )

        return result.model_dump()

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