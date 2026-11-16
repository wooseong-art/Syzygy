"""Build forecast files (submitted model, baseline, blends) from bt/*.json and score them with the
organisers' reference scorer (kit_src/.../scoring_demo/score.py --live --official-baselines)."""
import datetime as dt, glob, json, math, os, subprocess, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = r"c:\Users\min07\OneDrive\Desktop\solar_wind\Syzygy\submission_pack"
sys.path.insert(0, PACK)
from syzygy_live import ace, baseline            # our own stdlib-only code (same as in the image)
UTC = dt.timezone.utc
RT = os.path.join(HERE, "ace_rt"); FC = os.path.join(HERE, "bt_fc"); os.makedirs(RT, exist_ok=True); os.makedirs(FC, exist_ok=True)

docs = {}
for p in sorted(glob.glob(os.path.join(HERE, "bt", "*.json"))):
    docs[os.path.basename(p)[:-5]] = json.load(open(p, encoding="utf-8"))
t0s = sorted(docs)
parse = lambda s: dt.datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
first, last = parse(t0s[0]), parse(t0s[-1])

# ACE RT 1-min daily files for the baseline (same source and cut-off rule as the container)
samples = []
d = (first - dt.timedelta(days=30)).date()
while d <= last.date():
    f = os.path.join(RT, f"{d:%Y%m%d}.txt")
    if not os.path.exists(f):
        url = ace.RT_DAILY_URL.format(d=d)
        try:
            open(f, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "syzygy-backtest"}), timeout=60).read())
        except Exception as e:
            print("missing", d, e); open(f, "wb").write(b"")
    samples += ace.parse_rt_daily(open(f, encoding="utf-8", errors="replace").read())
    d += dt.timedelta(days=1)
samples.sort(key=lambda s: s[0])
print("cycles", len(t0s), t0s[0], "…", t0s[-1], "| RT samples", len(samples))

names = ["submitted", "model_KI_shift1", "model_KI_shift0", "baseline_tau48", "blend_exp6", "blend_exp12", "blend_exp24", "blend_exp48", "avg_50_50", "last1h_exp2", "last1h_exp3", "last1h_exp6", "last1h_exp12"]
out = {n: {} for n in names}
stats = {"stage": {}, "state": {}}
for t0s_ in t0s:
    doc, t0 = docs[t0s_], parse(t0s_)
    cutoff, start = t0 - ace.PUBLISH_LAG, (t0 - dt.timedelta(days=29)).replace(minute=0, second=0, microsecond=0)
    hourly = ace.hourly_means([s for s in samples if start <= s[0] <= cutoff], start, cutoff, ace.MIN_SAMPLES["RT"])
    base, _ = baseline.forecast(hourly, t0)
    res = ((doc.get("info") or {}).get("model") or {}).get("result") or {}
    hs = res.get("hourly_speed_kms")
    stats["stage"][doc["stage"]] = stats["stage"].get(doc["stage"], 0) + 1
    st = (res.get("status") or {}).get("state"); stats["state"][st] = stats["state"].get(st, 0) + 1
    out["submitted"][t0s_] = doc["forecast_speed_kms"]
    out["baseline_tau48"][t0s_] = base
    if hs and "ACE_48h" not in ((res.get("status") or {}).get("missing_inputs") or []):
        m1, m0 = hs[1:73], hs[0:72]
        out["model_KI_shift1"][t0s_], out["model_KI_shift0"][t0s_] = m1, m0
        for tau in (6, 12, 24, 48):
            out[f"blend_exp{tau}"][t0s_] = [math.exp(-(k + 1) / tau) * base[k] + (1 - math.exp(-(k + 1) / tau)) * m1[k] for k in range(72)]
        out["avg_50_50"][t0s_] = [0.5 * base[k] + 0.5 * m1[k] for k in range(72)]
        recent = [h for h in hourly if h >= t0 - dt.timedelta(hours=3)]
        plast = hourly[max(recent)] if recent else None       # last valid hourly mean before t0 (not the 6 h mean)
        for tau in (2, 3, 6, 12):
            out[f"last1h_exp{tau}"][t0s_] = m1 if plast is None else [math.exp(-(k + 1) / tau) * plast + (1 - math.exp(-(k + 1) / tau)) * m1[k] for k in range(72)]
    else:                                             # model unusable → what the container would upload (baseline)
        for n in names[1:]:
            out[n].setdefault(t0s_, base)
print("stages", stats)
files = []
for n in names:
    p = os.path.join(FC, n + ".json"); files.append(p)
    json.dump({"name": n, "forecasts": {k: {"forecast_speed_kms": [round(float(x), 2) for x in v]} for k, v in out[n].items()}}, open(p, "w"))
fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
demo = os.path.join(HERE, "kit_src", "benchlab-starter-kit", "scoring_demo")
rep = os.path.join(HERE, "bt_report.json")
r = subprocess.run([sys.executable, "score.py", *files, "--start", fmt(first), "--end", fmt(last), "--live", "--official-baselines", "-o", rep],
                   cwd=demo, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-3500:]); print(r.stderr[-1500:])
