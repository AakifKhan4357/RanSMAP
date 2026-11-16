from io import BytesIO
from pathlib import PurePosixPath
import stat
from typing import Optional
from zipfile import BadZipFile, ZipFile

from fastapi import HTTPException, UploadFile


STREAMS = {
    "ata_read": "ata_read.csv",
    "ata_write": "ata_write.csv",
    "mem_read": "mem_read.csv",
    "mem_write": "mem_write.csv",
    "mem_readwrite": "mem_readwrite.csv",
    "mem_exec": "mem_exec.csv",
}

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024


def _error(code: int, error_code: str, message: str, details=None):
    raise HTTPException(
        status_code=code,
        detail={
            "status": "error",
            "error_code": error_code,
            "message": message,
            "details": details or {},
        },
    )


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        _error(413, "UPLOAD_TOO_LARGE", "The upload exceeds the allowed size.")
    return data


def _read_zip(data: bytes) -> dict[str, bytes]:
    files = {}
    total_size = 0

    try:
        with ZipFile(BytesIO(data)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue

                raw_name = item.filename.replace("\\", "/")
                path = PurePosixPath(raw_name)

                if path.is_absolute() or ".." in path.parts:
                    _error(400, "UNSAFE_ARCHIVE_PATH", "The ZIP contains an unsafe path.")
                if raw_name.startswith("__MACOSX/"):
                    continue

                name = path.name
                if name not in STREAMS.values():
                    _error(415, "UNSUPPORTED_FILE", f"Unexpected file in ZIP: {name}")
                if name in files:
                    _error(400, "DUPLICATE_STREAM", f"The ZIP contains more than one {name}.")
                if stat.S_ISLNK(item.external_attr >> 16):
                    _error(400, "UNSAFE_ARCHIVE_PATH", "The ZIP contains a symbolic link.")

                total_size += item.file_size
                if total_size > MAX_UNZIPPED_BYTES:
                    _error(413, "UPLOAD_TOO_LARGE", f"The uncompressed ZIP is too large ({total_size / 1048576:.1f} MiB; limit {MAX_UNZIPPED_BYTES // 1048576} MiB).")

                files[name] = archive.read(item)

    except (BadZipFile, RuntimeError, NotImplementedError):
        _error(422, "CORRUPT_ARCHIVE", "The ZIP archive is invalid or corrupted.")

    missing = sorted(set(STREAMS.values()) - set(files))
    if missing:
        _error(
            400,
            "MISSING_STREAM",
            "The ZIP is missing required CSV streams.",
            {"missing_streams": missing},
        )

    return files


async def collect_stream_uploads(
    ata_read: Optional[UploadFile],
    ata_write: Optional[UploadFile],
    mem_read: Optional[UploadFile],
    mem_write: Optional[UploadFile],
    mem_readwrite: Optional[UploadFile],
    mem_exec: Optional[UploadFile],
    archive: Optional[UploadFile],
) -> dict[str, bytes]:
    if archive is not None and not archive.filename:
        archive = None  # Swagger can send a blank archive field.

    uploads = {
        "ata_read": ata_read,
        "ata_write": ata_write,
        "mem_read": mem_read,
        "mem_write": mem_write,
        "mem_readwrite": mem_readwrite,
        "mem_exec": mem_exec,
    }

    if archive is not None:
        if any(uploads.values()):
            _error(400, "MIXED_UPLOAD_METHODS", "Upload six CSVs or one ZIP, not both.")
        if not archive.filename.lower().endswith(".zip"):
            _error(415, "UNSUPPORTED_FORMAT", "The archive must be a .zip file.")
        return _read_zip(await _read_limited(archive, MAX_UPLOAD_BYTES))

    missing = [name for name, upload in uploads.items() if upload is None]
    if missing:
        _error(
            400,
            "MISSING_STREAM",
            "Upload all six CSV streams or one ZIP archive.",
            {"missing_streams": missing},
        )

    files = {}
    total_size = 0

    for field, upload in uploads.items():
        if not upload.filename.lower().endswith(".csv"):
            _error(415, "UNSUPPORTED_FORMAT", f"{field} must be a CSV file.")

        data = await _read_limited(upload, MAX_UPLOAD_BYTES)
        total_size += len(data)
        if total_size > MAX_UPLOAD_BYTES:
            _error(413, "UPLOAD_TOO_LARGE", f"The combined CSV upload is too large ({total_size / 1048576:.1f} MiB; limit {MAX_UPLOAD_BYTES // 1048576} MiB).")

        files[STREAMS[field]] = data

    return files