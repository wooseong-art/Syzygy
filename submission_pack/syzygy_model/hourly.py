"""Model output → the 72 hourly values Benchlab reads (forecast[j] is for t0 + (j + 1) h, j = 0…71).

Rule = the team's KI rule, computed inside fc_predict.py (git_repo 909af51, 2026-11-11):
  res["hourly_speed_kms"][k − 1] = ACE 1 h mean over [t0 + k − 1, t0 + k), k = 1…96
  = linear interpolation along lead time of v_now (placed at −3 h) and the four final 6 h means
    (placed at their window centres 21 / 45 / 69 / 93 h), read at k − 0.5 h.
This module only selects 72 of those 96 values; it does not change them.

Which 72: the scoring truth is SWPC ace_swepam_1h.json, whose time_tag marks the START of the averaging
hour (checked 2026-11-11: newest row at 15:40 UTC was 14:00; against the 1-min stream the MAE is 7.4 km/s
for [tag, tag + 1 h) vs 13.1 for [tag − 1 h, tag)). So the truth for "t0 + (j + 1) h" is the mean over
[t0 + j + 1, t0 + j + 2), which is the team's k = j + 2 → index j + 1.
LABEL_SHIFT_H = 1 selects hourly[1:73]; set it to 0 (hourly[0:72]) if the organisers say a forecast time
labels the END of its hour. syzygy_live/baseline.py uses the same start-of-hour convention.

Guards (failure tests 2026-11-10): no hourly output while the nets ran on an empty ACE history.
"""
import math

RULE = "KI (git_repo fc_predict.hourly): linear interp of v_now@-3h and final 6h means @21/45/69/93h"
LABEL_SHIFT_H = 1          # 1: forecast time = start of the truth hour (SWPC time_tag); 0: end of the hour
H = 72


def to_hourly(res, ace_hourly, t0):
    """res: fc_predict.predict() dict. Return (72 floats, info) or (None, reason)."""
    status = res.get("status") or {}
    if "ACE_48h" in (status.get("missing_inputs") or []):
        return None, "ACE_48h missing: model ran without solar-wind history"
    hs = res.get("hourly_speed_kms")
    if not isinstance(hs, list) or len(hs) < H + LABEL_SHIFT_H:
        return None, "model result has no hourly_speed_kms (model code older than git_repo 909af51?)"
    vals = [float(v) for v in hs[LABEL_SHIFT_H:LABEL_SHIFT_H + H]]
    if not all(math.isfinite(v) for v in vals):
        return None, "non-finite hourly values"
    return vals, dict(rule=RULE, label_shift_h=LABEL_SHIFT_H, hourly_model=res.get("hourly_model"),
                      final_model=res.get("final_model"), state=status.get("state"))
