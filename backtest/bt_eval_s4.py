"""Compare two backtest runs (same cycles, same replayed inputs) with the organisers' scorer.

    python bt_eval_s4.py <scoring_demo dir> <ace_monthly dir> runs_old runs_s4

1. Official score (score.py --live --official-baselines) of each run's uploaded forecast_speed_kms.
2. 6 h-mean errors at 24/48/72/96 h by situation (slow / fast / sudden rise / drop), from the monthly
   ACE SWEPAM 1 h files, status 0 only (same truth as the leaderboard).
3. Stage / model-state counts (fallbacks).
"""
import datetime as dt, glob, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
demo, ace_dir, *runs = sys.argv[1:]
FC = os.path.join(HERE, "forecasts_s4"); os.makedirs(FC, exist_ok=True)


def load(name):
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, name, "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p, encoding="utf-8"))
    return out


docs = {n: load(n) for n in runs}
common = sorted(set.intersection(*(set(d) for d in docs.values())))
print("cycles in common:", len(common), common[0], "...", common[-1])

files = []
for n in runs:
    st = {}
    for t0 in common:
        d = docs[n][t0]
        res = ((d.get("info") or {}).get("model") or {}).get("result") or {}
        k = f"{d['stage']} / {res.get('final_model')} / {(res.get('status') or {}).get('state')}"
        st[k] = st.get(k, 0) + 1
    print(f"[{n}] stages:", st)
    p = os.path.join(FC, n + ".json"); files.append(p)
    json.dump({"name": n, "forecasts": {t0: {"forecast_speed_kms": docs[n][t0]["forecast_speed_kms"]} for t0 in common}}, open(p, "w"))

fmt = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:00:00Z"
rep = os.path.join(HERE, f"report_s4_{common[0][:8]}_{common[-1][:8]}.json")
r = subprocess.run([sys.executable, "score.py", *files, "--start", fmt(common[0]), "--end", fmt(common[-1]),
                    "--live", "--official-baselines", "-o", rep], cwd=demo, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-2500:]); print(r.stderr[-800:])

# lead curve excerpt from the report
R = json.load(open(rep, encoding="utf-8"))
print("S_tau at lead 1/6/12/24/48/72 h  |  mean bias")
for name, sc in R["scores"].items():
    c = sc["s_curve"]
    print(f"  {name:26s}", [round(c[i - 1], 1) for i in (1, 6, 12, 24, 48, 72)], f"| bias {sc['mean_bias']:+.1f}")

# situation breakdown, 6 h means, status-0 truth
obs = {}
for f in glob.glob(os.path.join(ace_dir, "*.txt")):
    for ln in open(f):
        p = ln.split()
        if len(p) < 10 or not p[0].isdigit() or len(p[0]) != 4 or int(p[6]) != 0:
            continue
        v = float(p[8])
        if 200 < v < 1500:
            obs[dt.datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3][:2]))] = v
print("\n6 h means at 24/48/72/96 h, status-0 truth: RMSE (bias) km/s")
rows = {n: [] for n in runs}
for t0s in common:
    t0 = dt.datetime.strptime(t0s, "%Y%m%dT%H%M%SZ")
    for n in runs:
        res = docs[n][t0s]["info"]["model"]["result"]
        for j, h in enumerate((24, 48, 72, 96)):
            vals = [obs.get(t0 + dt.timedelta(hours=h - 6 + i)) for i in range(6)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                rows[n].append((np.mean(vals), res["v_now_kms"], res["final_speed_kms"][j], res["g01_kms"][j],
                                (res.get("router_cf_kms") or [np.nan] * 4)[j]))
for name, cond in (("all", lambda y, vn: np.ones_like(y, bool)), ("slow (obs < 500)", lambda y, vn: y < 500),
                   ("fast (obs >= 500)", lambda y, vn: y >= 500), ("rise >= +100", lambda y, vn: y - vn >= 100),
                   ("rise >= +200", lambda y, vn: y - vn >= 200), ("drop <= -100", lambda y, vn: y - vn <= -100)):
    line = f"{name:18s}"
    for n in runs:
        a = np.array(rows[n]); y, vn = a[:, 0], a[:, 1]; m = cond(y, vn)
        for lab, col in (("final", 2), ("G01", 3), ("Router", 4)):
            e = a[m, col] - y[m]
            line += f" | {n}:{lab} {np.sqrt(np.nanmean(e ** 2)):6.1f} ({np.nanmean(e):+6.1f})"
        line += f" n={int(m.sum())}"
    print(line)

# block bootstrap of S_final(run2) - S_final(run1), official metric, status-0 truth (obs[t0, tau] = v(t0 + tau))
sys.path.insert(0, demo)
from swp_scoring.core.metric import lead_score                # noqa: E402
T = [dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ") for s in common]
O = np.array([[obs.get(t + dt.timedelta(hours=tau), np.nan) for tau in range(1, 73)] for t in T])
P = {n: np.array([docs[n][s]["forecast_speed_kms"] for s in common], float) for n in runs}


def s_final(idx, n):
    vals = []
    for j in range(72):
        ls = lead_score(O[idx, j], P[n][idx, j])
        if ls is not None:
            vals.append(ls.s)
    return float(np.mean(vals))


full = np.arange(len(common))
base = {n: s_final(full, n) for n in runs}
print("\nS_final with monthly status-0 truth (own grid):", {n: round(v, 2) for n, v in base.items()})
BLOCK = 8                                                    # 2 days = 8 cycles per block
blocks = [full[i:i + BLOCK] for i in range(0, len(full), BLOCK)]
rng = np.random.default_rng(0)
d = []
for _ in range(500):
    idx = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
    d.append(s_final(idx, runs[1]) - s_final(idx, runs[0]))
d = np.array(d)
print(f"delta S_final {runs[1]} - {runs[0]} = {base[runs[1]] - base[runs[0]]:+.2f}, "
      f"2-day block bootstrap 95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}], "
      f"share of resamples where {runs[1]} is better: {(d < 0).mean():.0%}")
