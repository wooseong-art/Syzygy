import glob, json, os, subprocess, sys, numpy as np
RUNS = r"c:/Users/min07/OneDrive/Desktop/solar_wind/Syzygy/backtest/runs"
SP = os.path.dirname(os.path.abspath(__file__)); FC = os.path.join(SP, "blend_fc")
DEMO = os.path.join(SP, "kit_src", "benchlab-starter-kit", "scoring_demo")
ANCH = [-3, 21, 45, 69, 93]; X = np.arange(1, 97) - 0.5
ws = [round(x, 2) for x in np.arange(0, 1.0001, 0.1)]
out = {w: {} for w in ws}; maxdiff = 0
for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
    t0 = os.path.basename(p)[:-5]; d = json.load(open(p, encoding="utf-8"))
    r = d["info"]["model"]["result"]
    g, ro = np.array(r["g01_kms"]), np.array(r["router_cf_kms"])
    corr = np.array(r["correction_delta_kms"]); vnow = r["v_now_kms"]
    for w in ws:
        fin = (1 - w) * g + w * ro + corr                     # correction kept as in the run (approximation)
        hs = np.interp(X, ANCH, np.concatenate([[vnow], fin]))
        out[w][t0] = hs[1:73]
    maxdiff = max(maxdiff, float(np.max(np.abs(out[0.5][t0] - np.array(d["forecast_speed_kms"])))))
print("w=0.5 reproduces submitted: max |diff| =", round(maxdiff, 3), "km/s")
files = []
for w in ws:
    f = os.path.join(FC, f"w{w:.1f}.json"); files.append(f)
    json.dump({"name": f"w_router={w:.1f}", "forecasts": {k: {"forecast_speed_kms": [round(float(x), 2) for x in v]} for k, v in out[w].items()}}, open(f, "w"))
ks = sorted(out[0.5]); fmt = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:00:00Z"
rep = os.path.join(SP, "blend_report.json")
res = subprocess.run([sys.executable, "score.py", *files, "--start", fmt(ks[0]), "--end", fmt(ks[-1]), "--live", "-o", rep],
                     cwd=DEMO, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(res.stdout[-3000:]); print(res.stderr[-800:])
