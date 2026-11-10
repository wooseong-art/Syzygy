"""Model output (6 h means at h = 24/48/72/96, plus 16-bin Router/RegimeNet when images were available)
→ 72 hourly values for t0+1h … t0+72h.

NOT DECIDED YET (CLAUDE.md §9 단계 C-4): the rule will be chosen by backtest on the team's archive
predictions. Until then to_hourly() returns None, so predict.py keeps the baseline forecast and only
records the model output in the output file's "info.model".

Guards to keep when the rule is written (seen in failure tests, 2026-11-10):
  - res["status"]["missing_inputs"] contains "ACE_48h" → return None (the nets ran on an empty ACE history)
  - res["status"]["state"] == "FALLBACK_G01" → G01 only (no Router 16 bins); still usable if ACE is present
"""
RULE = None


def to_hourly(res, ace_hourly, t0):
    """res: fc_predict.predict() dict. Return 72 floats or None."""
    return None
