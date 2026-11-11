#!/usr/bin/env python3
"""Real-submission helper for submission_pack/ (the conformance one is submit.py).

  python submit_real.py check            # default: verify the pack and build the zip. Uploads nothing.
  python submit_real.py status           # eligibility, granted resources, recent runs, log links
  python submit_real.py watch [minutes]  # poll status every 60 s and print changes (default 40 min)
  python submit_real.py logs             # download the build / run logs into submission_logs/
  python submit_real.py submit --yes     # check, zip, UPLOAD. Replaces the current submission.

Same upload path as submit.py: GET the portal page (PORTAL in .team) -> presigned S3 POST form -> upload.
A new upload replaces the previous submission and must pass build + dry run on its own to be ELIGIBLE.
"""
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(HERE, "submission_pack")
ZIP = os.path.join(HERE, "submission_real.zip")            # matched by .gitignore (submission_*.zip)
LOGDIR = os.path.join(HERE, "submission_logs")
SKIP_DIRS = {"__pycache__", "out", ".ipynb_checkpoints"}
SKIP_EXT = (".pyc", ".zip", ".log")
REQUIRED = ("Dockerfile", "submission.json", "README.md", "run.sh", "upload.sh", "predict.py", "model/MANIFEST.json")
FARGATE = {1024: (2048, 8192, 1024), 2048: (4096, 16384, 1024), 4096: (8192, 30720, 1024), 8192: (16384, 32768, 4096)}


def die(msg):
    print("ERROR: %s" % msg, file=sys.stderr)
    sys.exit(1)


def portal_url():
    path = os.path.join(HERE, ".team")
    if not os.path.exists(path):
        die(".team not found")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            k, _, v = line.strip().partition("=")
            if k.strip() == "PORTAL":
                v = v.strip().strip('"').strip("'")
                if v.startswith("http"):
                    return v
    die("no usable PORTAL line in .team")


def fetch_portal():
    try:
        with urllib.request.urlopen(portal_url(), timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        die("portal returned HTTP %d (403 = PORTAL link wrong or cut off)" % e.code)
    except Exception as e:                                   # noqa: BLE001
        die("could not reach the portal: %s" % e)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack_files():
    out = []
    for root, dirs, names in os.walk(PACK):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(SKIP_EXT):
                continue
            full = os.path.join(root, name)
            out.append((os.path.relpath(full, PACK).replace(os.sep, "/"), full))
    return sorted(out)


def check():
    """Verify the pack and write the zip. Returns the submission.json dict."""
    files = dict(pack_files())
    missing = [f for f in REQUIRED if f not in files]
    if missing:
        die("missing in submission_pack/: %s" % ", ".join(missing))
    with open(files["submission.json"], encoding="utf-8") as fh:
        sub = json.load(fh)
    cpu, mem = int(sub["cpu_units"]), int(sub["memory_mib"])
    if cpu not in FARGATE or not (FARGATE[cpu][0] <= mem <= FARGATE[cpu][1]) or mem % FARGATE[cpu][2]:
        die("cpu_units %s / memory_mib %s is not a valid Fargate combination" % (cpu, mem))
    print("submission.json: team %s, %d CPU units, %d MiB, domains %s" % (sub["team_name"], cpu, mem, sub["declared_domains"]))

    with open(files["model/MANIFEST.json"], encoding="utf-8") as fh:
        man = json.load(fh)
    bad = [e["path"] for e in man["files"]
           if "model/" + e["path"] not in files or sha256(files["model/" + e["path"]]) != e["sha256"]]
    if bad:
        die("model/ does not match MANIFEST (run tools/sync_model.py): %s" % ", ".join(bad[:5]))
    print("model/: %d files match MANIFEST, source commit %s" % (len(man["files"]), man.get("commit", "?")[:7]))

    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, full in files.items():
            zf.write(full, rel)
    size = os.path.getsize(ZIP)
    print("%s: %d files, %.1f MB, sha256 %s" % (os.path.basename(ZIP), len(files), size / 1e6, sha256(ZIP)[:16]))
    if size > 500 * 1024 * 1024:
        die("zip is larger than the 500 MB upload limit")
    return sub


def upload(form):
    url, fields = form["url"], form["fields"]
    bnd = "----py" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():                              # keep the portal's own order
        body += b"--" + bnd.encode() + b"\r\n"
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % k).encode() + v.encode() + b"\r\n"
    with open(ZIP, "rb") as fh:                              # 'file' MUST come last: S3 ignores anything after it
        data = fh.read()
    body += b"--" + bnd.encode() + b"\r\n"
    body += b'Content-Disposition: form-data; name="file"; filename="submission.zip"\r\n'
    body += b"Content-Type: application/zip\r\n\r\n" + data + b"\r\n--" + bnd.encode() + b"--\r\n"
    req = urllib.request.Request(url, data=bytes(body), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + bnd})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            print("HTTP %d - the zip arrived (%s UTC)." % (r.status, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
    except urllib.error.HTTPError as e:
        print(e.read().decode("utf-8", "replace")[:900], file=sys.stderr)
        die("HTTP %d - upload rejected. The form lasts 1 hour; run again." % e.code)


def summary(doc):
    runs = doc.get("recent_runs") or []
    return {"eligibility": doc.get("eligibility"), "phase": doc.get("phase"), "accepting": doc.get("accepting"),
            "failure_reason": doc.get("failure_reason"),
            "cpu_granted": doc.get("cpu_granted"), "memory_granted": doc.get("memory_granted"),
            "cpu_requested": doc.get("cpu_requested"), "memory_requested": doc.get("memory_requested"),
            "logs": ["%s (%s bytes)" % (x.get("key", "").split("/", 1)[-1], x.get("size")) for x in doc.get("logs") or []],
            "recent_runs": [{k: r.get(k) for k in ("cycle_ts", "status", "exit_code", "pull_seconds", "run_seconds",
                                                    "output_bytes", "failure_reason", "stopped_reason")} for r in runs]}


def get_logs(doc):
    os.makedirs(LOGDIR, exist_ok=True)
    for x in doc.get("logs") or []:
        name = x.get("key", "log").split("/", 1)[-1].replace("/", "_")
        try:
            with urllib.request.urlopen(x["url"], timeout=60) as r:
                data = r.read()
        except Exception as e:                               # noqa: BLE001
            print("could not download %s: %s" % (name, e))
            continue
        if name.endswith(".gz"):
            try:
                data, name = gzip.decompress(data), name[:-3] + ".log"
            except OSError:
                pass
        with open(os.path.join(LOGDIR, name), "wb") as fh:
            fh.write(data)
        print("saved submission_logs/%s (%d bytes)" % (name, len(data)))


def watch(minutes):
    last, end = None, time.time() + minutes * 60
    while time.time() < end:
        s = summary(fetch_portal())
        if s != last:
            print(time.strftime("[%H:%M:%S UTC]", time.gmtime()), json.dumps(s, indent=1))
            last = s
        if s["eligibility"] == "ELIGIBLE":
            print("ELIGIBLE.")
            return
        time.sleep(60)
    print("watch ended after %d minutes; eligibility = %s" % (minutes, last and last["eligibility"]))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check()
        print("\ncheck only - nothing uploaded.")
    elif cmd == "status":
        print(json.dumps(summary(fetch_portal()), indent=1))
    elif cmd == "watch":
        watch(float(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "logs":
        get_logs(fetch_portal())
    elif cmd == "submit":
        if "--yes" not in sys.argv:
            die("this REPLACES the current submission. Re-run as:  python submit_real.py submit --yes")
        check()
        doc = fetch_portal()
        if doc.get("accepting") != "your real submission":
            die("portal is not accepting a real submission (accepting = %r)" % doc.get("accepting"))
        form = doc.get("upload")
        if not form or not form.get("url") or not form.get("fields"):
            die("the portal did not return an upload form")
        print("\nUploading...")
        upload(form)
        print("Build + dry run follow. Track with:  python submit_real.py watch")
    else:
        die(__doc__)


if __name__ == "__main__":
    main()
