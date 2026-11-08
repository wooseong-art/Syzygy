"""Model stage — placeholder until the trained Syzygy model (syzygy_code/fc_predict.py,
its original config.py and checkpoints) is wired in.

Contract for the real implementation:
    predict(t0, ace_hourly, deadline_monotonic) -> list of 72 floats (km/s) or None
      t0          : timezone-aware UTC datetime (the platform's inference timestamp)
      ace_hourly  : {hour_start (aware UTC datetime): V km/s}, already cut off at t0
      deadline    : time.monotonic() value by which the stage must return
    Return None (or raise) to fall back to the baseline forecast.
    Fetch SDO images via syzygy_live.sdo (find_aia / find_hmi / download), which enforce the t0 cut-off.
"""
MODEL_AVAILABLE = False


def predict(t0, ace_hourly, deadline_monotonic):
    return None
