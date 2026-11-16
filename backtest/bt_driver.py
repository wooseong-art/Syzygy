"""Backtest driver: run the submitted image (swp-local) for past cycles and keep each output JSON.
   python bt_driver.py <first YYYYMMDD> <last YYYYMMDD> [parallel]"""
import datetime as dt, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt")
a, b = (dt.datetime.strptime(x, "%Y%m%d") for x in sys.argv[1:3])
par = int(sys.argv[3]) if len(sys.argv) > 3 else 2
cycles = []
t = a
while t <= b + dt.timedelta(hours=18):
    cycles.append(t.strftime("%Y%m%dT%H0000Z")); t += dt.timedelta(hours=6)
cycles.reverse()                                    # newest first (NRT files still exist there)
KEEP = '"event": "(prefetch_done|predict_done|model_done|model_failed|stage_skipped|stage_failed|ace_summary|watchdog_exit)"'
def run(t0):
    path = os.path.join(OUT, t0 + ".json")
    if os.path.exists(path): return t0, "cached", 0
    s = time.time()
    cmd = ["docker", "run", "--rm", "--cpus=4", "--memory=8g", "-e", "SYZYGY_DL_WORKERS=8", "--entrypoint", "bash", "swp-local", "-c",
           f"python3 /app/predict.py {t0} /tmp/f.json > /tmp/log 2>&1; cat /tmp/f.json; echo; echo ===LOG; grep -E '{KEEP}' /tmp/log; grep -c cutoff_filter /tmp/log; grep -c '\"status\": 503' /tmp/log"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, encoding="utf-8", errors="replace")
        body, _, log = p.stdout.partition("===LOG")
        doc = json.loads(body)
        doc["_log"] = log.strip().splitlines()
        with open(path + ".tmp", "w", encoding="utf-8") as f: json.dump(doc, f)
        os.replace(path + ".tmp", path)
        st = (doc.get("info", {}).get("model") or {}).get("result", {}).get("status", {}).get("state")
        return t0, f"{doc['stage']} {st}", time.time() - s
    except Exception as e:
        return t0, "ERROR " + repr(e)[:200], time.time() - s
with ThreadPoolExecutor(par) as ex:
    for i, (t0, msg, sec) in enumerate(ex.map(run, cycles)):
        print(f"{i + 1}/{len(cycles)} {t0} {msg} {sec:.0f}s", flush=True)
print("DONE", flush=True)
