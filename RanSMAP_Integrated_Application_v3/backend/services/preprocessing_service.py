
from typing import Dict, Optional

from backend.preprocessing.pipeline import process_files
from backend.preprocessing.config import PreprocessResponse


def run_preprocessing(
    files: Dict[str, bytes],
    sample_id: str = "test_sample",
    label: Optional[str] = None
) -> PreprocessResponse:

    result = process_files(
        file_inputs=files,
        sample_id=sample_id,
        label=label,
        t_d_sec=30.0,
        t_window_sec=0.1,
        step_epsilon_sec=0.1,
        raise_on_rejection=True
    )

    return result
