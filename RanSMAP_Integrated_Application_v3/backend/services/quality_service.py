"""Input-quality diagnostics for the (300, 23) feature matrix."""
import numpy as np

WINDOW_SECONDS = 0.1
# The CNN-LSTM reads the last LSTM state, so it mostly "sees" the final ~8-10 s.
# Activity that stops before window 220 (22 s) is effectively invisible to it.
LATE_ACTIVITY_WINDOW = 220


def capture_diagnostics(features) -> dict:
    arr = np.asarray(features)
    total = int(arr.shape[0]) if arr.ndim == 2 else 0
    if total == 0:
        return {"active_windows": 0, "last_active_window": None, "warnings": []}

    active = (arr != 0).any(axis=1)
    active_windows = int(active.sum())
    warnings = []

    if active_windows == 0:
        last = None
        warnings.append("No activity was found in any window; the result is not meaningful.")
    else:
        last = int(np.nonzero(active)[0].max())
        if last < LATE_ACTIVITY_WINDOW:
            seconds = (last + 1) * WINDOW_SECONDS
            warnings.append(
                f"Capture activity ends at {seconds:.1f} s of the {total * WINDOW_SECONDS:.0f} s window "
                f"({active_windows} of {total} windows contain data). The model relies mostly on the "
                "last ~8 s of the window, so a Benign result for this short capture may be unreliable."
            )

    return {"active_windows": active_windows, "last_active_window": last, "warnings": warnings}
