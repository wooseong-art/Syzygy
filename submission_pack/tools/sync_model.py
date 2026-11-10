#!/usr/bin/env python3
"""Copy the team's inference code + weights from a git_repo checkout into submission_pack/model/.

    python tools/sync_model.py <path to wooseong-art/git_repo checkout>

Files are copied byte-for-byte (never edited) at the same relative paths as in git_repo, so
fc_common.py's path logic (EXP, CODE = EXP.parents[1], config BASE_DIR = repo root) keeps working.
All Docker-specific changes live in syzygy_model/ (runtime patches), not here.

What is copied
  - code: the static import closure of fc_predict.py (+ the PFSS / CH modules it imports lazily)
  - data: the json / pkl / pt files Pipeline() reads, discovered from the selection and info json files
model/MANIFEST.json records source commit, path, size and sha256. Re-running prints which files were
added / changed / removed, so a model update only needs this script + a rebuild.
"""
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / "model"
EXP_FINAL = "code/experiments/syzygy_final_cme_free_2026"
COLLIN = "code/experiments/collin_hss_reference"
HSSON = "code/experiments/hss_onset_conditional_calibration"
PHYS = "code/experiments/physical_factorial_ablation"
SEARCH = [EXP_FINAL, HSSON, COLLIN, PHYS, "code"]          # sys.path order used by fc_common / oc_common / cr_common
ENTRY = ["fc_predict", "04m_ch_pfss", "04f_pfss_maps", "04b_ch_features", "pfss", "pfss_trace", "build_features"]
SEEDS = (42, 43, 44, 45, 46)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_module(src, name):
    for d in SEARCH:
        for rel in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
            if (src / d / rel).exists():
                return f"{d}/{rel}"
    return None


def code_closure(src):
    seen, todo = {}, list(ENTRY)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        rel = find_module(src, m)
        if rel is None:
            continue                                           # third-party / stdlib
        seen[m] = rel
        if "." in m:                                           # package: keep its __init__ too
            pkg = find_module(src, m.rsplit(".", 1)[0])
            if pkg:
                seen[m.rsplit(".", 1)[0]] = pkg
        for n in ast.walk(ast.parse((src / rel).read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                todo.append(n.module)
            elif (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "import_module"
                  and n.args and isinstance(n.args[0], ast.Constant)):
                todo.append(n.args[0].value)
    return sorted(set(seen.values()))


PROSWIN_SRC = "external/proswin/src"
PROSWIN_ENTRY = ["proswin.evaluation.hss_finder", "proswin.evaluation.solar_wind_prediction_evaluator", "proswin.predictions",
                 "proswin.utils.interpolation_dates", "proswin.model.losses", "proswin.data_management.event_builder"]


def proswin_files(src):
    """cr_events.py calls cr_common.load_proswin() at import time → these external (MIT) modules are needed."""
    base, seen, todo = src / PROSWIN_SRC, set(), list(PROSWIN_ENTRY)
    while todo:
        m = todo.pop()
        parts = m.split(".")
        for i in range(1, len(parts)):                         # package __init__ files
            init = "/".join(parts[:i]) + "/__init__.py"
            if (base / init).exists():
                seen.add(init)
        rel = m.replace(".", "/") + ".py"
        if rel in seen or not (base / rel).exists():
            continue
        seen.add(rel)
        for n in ast.walk(ast.parse((base / rel).read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names if a.name.startswith("proswin")]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.startswith("proswin"):
                todo.append(n.module)
    return sorted(f"{PROSWIN_SRC}/{r}" for r in seen) + ["external/proswin/LICENSE"]


def local(p):
    """Absolute Windows path from a training json (D:\\syzygy\\code\\...) → repo-relative 'code/...'."""
    s = str(p).replace("\\", "/")
    i = s.find("/code/")
    if i < 0:
        raise ValueError(f"cannot map {p}")
    return "code/" + s[i + len("/code/"):]


def data_files(src):
    rd = lambda rel: json.loads((src / rel).read_text(encoding="utf-8"))
    sel = rd(f"{EXP_FINAL}/configs/final_selection.json")
    variant = sel["router_variant"]
    out = [f"{EXP_FINAL}/configs/final_selection.json",
           f"{EXP_FINAL}/metrics/ensemble_meta_valid.json",
           f"{EXP_FINAL}/preregistration.json",
           f"{EXP_FINAL}/final_bundle/final_params.py",
           f"{COLLIN}/predictions/preds_info.json",
           f"{HSSON}/predictions/model_preds_info.json"]
    out += [str(p.relative_to(src)).replace("\\", "/") for p in sorted((src / EXP_FINAL / "configs").glob("boxcox_maps_*.pkl"))]
    info = rd(f"{COLLIN}/predictions/preds_info.json")
    for s in SEEDS:
        out.append(f"{EXP_FINAL}/checkpoints/regime/regime_cf_s{s}.pt")
        meta = f"{EXP_FINAL}/checkpoints/router/meta_router_{variant}_s{s}.json"
        out += [meta, local(rd(meta)["checkpoint"])]
        out += [local(info[f"{k}|{s}"]["ckpt"]) for k in ("G4x3", "P2o_4x3")]
        out.append(f"{HSSON}/checkpoints/O_seed{s}/best.pt")
        out.append(f"{HSSON}/checkpoints/{'GRID_T2_lam0.1_beta0.0_seed42' if s == 42 else f'T2_seed{s}'}/best.pt")
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).resolve()
    files = code_closure(src) + proswin_files(src) + data_files(src)
    missing = [f for f in files if not (src / f).exists()]
    if missing:
        sys.exit("missing in source repo:\n  " + "\n  ".join(missing))
    old = {}
    if (DEST / "MANIFEST.json").exists():
        old = {e["path"]: e["sha256"] for e in json.loads((DEST / "MANIFEST.json").read_text(encoding="utf-8"))["files"]}
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    entries = []
    for f in files:
        d = DEST / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, d)
        entries.append({"path": f, "bytes": d.stat().st_size, "sha256": sha256(d)})
    new = {e["path"]: e["sha256"] for e in entries}
    for f in set(old) - set(new):
        (DEST / f).unlink(missing_ok=True)
    (DEST / "MANIFEST.json").write_text(json.dumps({"source": "https://github.com/wooseong-art/git_repo", "commit": commit,
                                                   "files": entries}, indent=1), encoding="utf-8")
    added = sorted(set(new) - set(old))
    changed = sorted(f for f in set(new) & set(old) if new[f] != old[f])
    removed = sorted(set(old) - set(new))
    print(f"commit {commit or '?'}: {len(entries)} files, {sum(e['bytes'] for e in entries) / 1e6:.1f} MB")
    for tag, lst in (("added", added if old else []), ("changed", changed), ("removed", removed)):
        for f in lst:
            print(f"  {tag:8s} {f}")
    if old and not (added or changed or removed):
        print("  no changes")


if __name__ == "__main__":
    main()
