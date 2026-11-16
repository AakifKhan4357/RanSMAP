import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.health import router as health_router
from backend.routes.preprocess import router as preprocess_router
from backend.routes.predict import router as predict_router
from backend.routes.model_status import router as model_status_router
from backend.routes.compare import router as compare_router
from backend.routes.adaptive import router as adaptive_router


logger = logging.getLogger(__name__)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "RANSMAP_CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="RanSMAP API",
    description="Ransomware Detection using Storage and Memory Behavioral Analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException):
    detail = exc.detail

    if isinstance(detail, dict) and detail.get("status") == "error":
        body = detail
    else:
        body = {
            "status": "error",
            "error_code": "REQUEST_FAILED",
            "message": "The request could not be completed.",
            "details": detail if isinstance(detail, dict) else {},
        }

    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_code": "INVALID_REQUEST",
            "message": "The request fields are invalid.",
            "details": {"validation_errors": errors},
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unexpected API error")

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected server error occurred.",
            "details": {},
        },
    )


app.include_router(health_router, prefix="/api")
app.include_router(preprocess_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(model_status_router, prefix="/api")
app.include_router(compare_router, prefix="/api")
app.include_router(adaptive_router, prefix="/api")


@app.get("/")
def home():
    return {
        "project": "RanSMAP",
        "message": "RanSMAP API is running",
    }