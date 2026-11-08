"""Fallback forecast from ACE history alone: persistence blended into 27-day recurrence.

  v(t0 + k h) = w_k * v_now + (1 - w_k) * v_rec(t0 + k h),   w_k = exp(-k / TAU_H),  k = 1..72
    v_now : mean of valid hourly V in [t0 - 6 h, t0)            (else the last valid hour within 48 h)
    v_rec : mean of valid hourly V within +-REC_HALF_H around (t0 + k h - SYNODIC_H)
Missing pieces degrade gracefully: no recurrence -> persistence; no persistence -> recurrence;
neither -> climatology.
"""
import math
from datetime import timedelta

SYNODIC_H = 655          # 27.2753 d solar synodic rotation, rounded to whole hours
REC_HALF_H = 3           # recurrence smoothing half-width (h)
TAU_H = 48.0             # persistence e-folding time (h). Backtest 2026-06..11, 512 t0s, ACE RT truth:
                         # MAE persistence 70.2 / recurrence 64.1 / tau 24h 52.2 / tau 48h 51.0 / tau 96h 55.1 km/s
CLIMATOLOGY_KMS = 430.0  # same as the model's V_MEAN
H = 72


def _mean(hourly, hours):
    vals = [hourly[h] for h in hours if h in hourly]
    return sum(vals) / len(vals) if vals else None


def v_now(hourly, t0):
    t0h = t0.replace(minute=0, second=0, microsecond=0)
    v = _mean(hourly, [t0h - timedelta(hours=i) for i in range(1, 7)])
    if v is not None:
        return v, "mean_6h"
    for i in range(7, 49):
        h = t0h - timedelta(hours=i)
        if h in hourly:
            return hourly[h], f"last_valid_{i}h_ago"
    return None, "none"


def forecast(hourly, t0, tau_h=TAU_H):
    """Return (72 values, info dict)."""
    t0h = t0.replace(minute=0, second=0, microsecond=0)
    now, now_src = v_now(hourly, t0)
    out, n_rec = [], 0
    for k in range(1, H + 1):
        c = t0h + timedelta(hours=k - SYNODIC_H)
        rec = _mean(hourly, [c + timedelta(hours=d) for d in range(-REC_HALF_H, REC_HALF_H + 1)])
        n_rec += rec is not None
        if now is not None and rec is not None:
            w = math.exp(-k / tau_h)
            out.append(w * now + (1 - w) * rec)
        elif now is not None:
            out.append(now)
        elif rec is not None:
            out.append(rec)
        else:
            out.append(CLIMATOLOGY_KMS)
    kind = ("persistence+recurrence" if now is not None and n_rec else
            "persistence" if now is not None else "recurrence" if n_rec else "climatology")
    return out, dict(kind=kind, v_now=None if now is None else round(now, 1), v_now_source=now_src,
                     recurrence_hours=n_rec, tau_h=tau_h)
