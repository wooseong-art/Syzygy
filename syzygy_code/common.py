"""
common.py — physical factorial ablation 공통 상수·경로·도구.

실행 위치: D:\\syzygy\\code\\experiments\\physical_factorial_ablation (코드 + 출력).
OneDrive 의 solarwind/experiments/physical_factorial_ablation 은 코드·문서 원본 (출력은 D: 에만 — OneDrive 동기화가
학습 중 계속 쓰는 events.jsonl·checkpoint 를 잠그지 않도록).
"""
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]                       # solarwind 코드 루트 (config.py, sw_utils.py)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))              # 뒤에 붙여 실험 폴더 모듈이 우선 (이름 충돌 방지)

import config as C                          # noqa: E402

for _s in (sys.stdout, sys.stderr):           # 출력이 파일·파이프여도 한글·기호에서 죽지 않게 (Windows cp949)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DIRS = {k: EXP / k for k in ("configs", "checkpoints", "features", "predictions", "metrics", "figures", "monitoring", "reports")}
FEATURES_FILE = DIRS["features"] / "features.npz"
FEATURES_META = DIRS["features"] / "features_meta.json"

HORIZONS = (24, 48, 72, 96)                 # 예측 horizon (h)
KH = tuple(h // C.BIN_HOURS - 1 for h in HORIZONS)   # dataset target bin 번호 (bin = [t+h-6h, t+h) 6 h 평균) → 3, 7, 11, 15
V_MEAN, V_STD = C.V_MEAN, C.V_STD           # 속도 정규화 (430, 100) — 프로젝트 공통 고정 상수

# 공통 과거 속도 입력
HIST_HOURS = 48                             # t 직전 48 h hourly
NOW_HOURS = 6                               # 현재 속도 = 직전 6 h 평균 (RegimeNet anchor 와 같은 정의)
DAY27_H = 27 * 24                           # 정확히 27 일 = 648 h (사용자 지시; 기존 recurrence 는 655 h = 27.27 일)
REC_START_H = 28 * 24                       # 26–28 일 주변 창: [t − 28 d, t − 22 d) 6 h bin 24 개 (모든 horizon 의 h−28d … h−26d 포함)
REC_BINS = 24
V27_LEADS = (0, 24, 48, 72, 96)             # v_27(t, h) = 6 h 평균 [t+h−648−6, t+h−648) — REC 창의 (h+18)/6 번째 bin
V27_REC_IDX = tuple((h + 18) // 6 for h in V27_LEADS)

# B: 코로나홀별 signed 평균 Br
BR_MIN_MU = C.CH_MIN_MU                     # Br ≈ B_LOS / μ, μ ≥ 0.25 (CH 검출 가능 영역과 같음)
# G: 위도 4 × 경도 3 격자 (Collin+ 2025 의 equal-extent 4×3: 위도 45°, 경도 60° 칸; Stonyhurst 위경도, B0 보정)
G_LAT_EDGES = (-90.0, -45.0, 0.0, 45.0, 90.0)        # 행: 남 → 북 (A1–A3 = 가장 북쪽 행이 되도록 저장 시 뒤집음)
G_LON_EDGES = (-90.0, -30.0, 30.0, 90.0)             # 열: 동(−) → 서(+)
G_LAGS = (0, 24, 48, 72)                    # t, t−24 h, t−48 h, t−72 h
# C: CME
CME_WINDOW_H = 120                          # 발행 시점 이전 120 h
CME_LAG_H = C.CME_REPORT_LAG_HOURS          # time21_5 + 6 h ≤ t 이어야 '관측 가능'

GROUPS = ("B", "C", "G")
FACTORIAL_IDS = ("E000", "E100", "E010", "E001", "E110", "E101", "E011", "E111")


def groups_of(fid: str) -> dict:
    return {g: fid[1 + i] == "1" for i, g in enumerate(GROUPS)}


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def utcstamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, obj, indent=2):
    """임시 파일에 쓰고 fsync 뒤 os.replace (읽는 쪽은 항상 완전한 파일만 본다)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
        f.flush()
        os.fsync(f.fileno())
    for _ in range(50):                      # Windows: 읽는 프로세스가 잠깐 열고 있으면 replace 가 실패할 수 있음
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            import time
            time.sleep(0.05)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def code_version() -> dict:
    """git 저장소가 아니므로 commit hash 대신 실험 코드·공용 코드 파일의 SHA-256 을 기록한다."""
    files = sorted(EXP.glob("*.py")) + sorted((EXP / "monitoring").glob("*.py")) + [ROOT / "config.py", ROOT / "sw_utils.py"]
    h = hashlib.sha256()
    per = {}
    for f in files:
        if f.exists():
            b = f.read_bytes()
            per[str(f.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()[:16]
            h.update(b)
    git = None
    try:
        import subprocess
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        pass
    return {"git_commit": git, "git_note": None if git else "not a git repository — code SHA-256 recorded instead",
            "code_sha256": h.hexdigest()[:16], "files": per}


def env_info() -> dict:
    import numpy
    import torch
    info = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy": numpy.__version__, "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "device": "cpu"}
    try:
        import torch_directml
        info["directml_device"] = torch_directml.device_name(0)
    except Exception:
        info["directml_device"] = None
    return info


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def set_output_root(root: Path):
    """smoke test 등: features 를 뺀 출력 폴더를 root 아래로 돌린다 (실제 monitoring/checkpoint 를 건드리지 않게)."""
    root = Path(root)
    for k in DIRS:
        if k != "features":
            DIRS[k] = root / k
    ensure_dirs()
