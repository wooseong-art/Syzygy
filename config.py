"""
config.py
-----------------------------------
파이프라인 전역에서 공유하는 설정.

[디스크 분리]
  D:\flare_data\params\  ← Stage 1 (params) 기존 저장본, 그대로 사용
  E:\flare_data\         ← Stage 2 (PIL) 부터 모든 출력

처음 한 번만 D: 의 작은 메타 파일들을 E: 로 복사해두면 됨 (CMD):
    mkdir E:\flare_data
    xcopy D:\flare_data\*.* E:\flare_data\ /Y
  (params\ 서브폴더는 안 따라감. 의도된 동작.)
"""
from pathlib import Path

# ───────── 1. 디렉터리 ─────────
BASE_DIR        = Path(r"E:\flare_data")         # ← 출력 (PIL 이후 전부 여기)
RAW_DIR         = Path(r"E:\data")               # JSOC FITS (기존 그대로)
PARAM_DIR       = Path(r"D:\flare_data\params")  # ← D: 유지 (이미 저장됨)

HDF5_PATH       = BASE_DIR / "dataset.h5"        # = HDF5_PATH_MASKED (호환)
LABEL_CSV       = BASE_DIR / "labels.csv"
SPLIT_CSV       = BASE_DIR / "splits.csv"
CKPT_DIR        = BASE_DIR / "checkpoints"
LOG_DIR         = BASE_DIR / "logs"
HARP_NOAA_MAP   = BASE_DIR / "all_harps_with_noaa_ars.txt"
PIL_DIR         = BASE_DIR / "pil_images"

# ── 두 가지 변형 데이터셋 (masked vs full unmasked) ────────────────
HDF5_PATH_MASKED = HDF5_PATH                     # = dataset.h5
HDF5_PATH_FULL   = BASE_DIR / "dataset_full.h5"
RUNS_DIR         = BASE_DIR / "runs"             # runs/masked, runs/full

# PARAM_DIR 은 D: 의 기존 데이터이므로 자동 생성 대상에서 제외.
# (혹시 D: 가 마운트 안 돼 있을 때 빈 디렉터리를 만드는 사고 방지)
for d in [RAW_DIR, PIL_DIR, CKPT_DIR, LOG_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ───────── 2. JSOC ─────────
JSOC_EMAIL    = "wooseong@khu.ac.kr"     # 02b 의 drms.Client 에 사용. JSOC 등록 이메일과 일치해야 함.
SHARP_SERIES  = "hmi.sharp_cea_720s"     # 02b 의 recordset query 에 사용
# DATE_START/END 는 04_label_goes_hek.py 가 params/ 에서 자동 도출하므로 제거됨

# ───────── 3. 10채널 파라미터 이름 ─────────
PARAM_CHANNELS = [
    "Br",    # 0: radial field B_r (= B_z in CEA)
    "GAM",   # 1: gamma 경각(deg) = arctan(B_h / |B_z|)
    "GBT",   # 2: |∇ B_total|
    "GBH",   # 3: |∇ B_h|
    "GBZ",   # 4: |∇ B_z|
    "SJZ",   # 5: 수직 전류밀도 J_z precursor (Gauss/pixel)
    "ASJH",  # 6: current helicity proxy B_z · J_z
    "POT",   # 7: 광구 잉여 자기에너지 밀도
    "SHR",   # 8: 3D shear 각도(deg)
    "ALPHA", # 9: twist α = J_z / B_z
]
NUM_CHANNELS = len(PARAM_CHANNELS)

# ───────── 4. 이미지 사이즈 / 패치 ─────────
TARGET_H, TARGET_W = 224, 224
POTENTIAL_PAD_FACTOR = 2     # FFT padding for potential-field channels; 2 keeps original behavior
MAX_VIEW_ANGLE_DEG = 70.0   # heliocentric viewing angle cutoff; samples above this are excluded
PIL_DILATION       = 8
PIL_BZ_THRESHOLD_NORM  = 0.2     # (구) 정규화 후 임계값 — 현재 02_pil_image.py 는 사용 안 함
PIL_BZ_THRESHOLD_GAUSS = 200.0   # (신) raw Br Gauss 임계값. 모든 sample 동일. Schrijver 150 G ~ Falconer 100 G 사이.
PIL_GRAD_THRESHOLD = 50.0
MIN_PIL_PIXELS     = 30

CONV_PIL_SIGMA        = 5.0      # Gaussian σ (px)
CONV_PIL_MIN_STRENGTH = 0.001    # 임계값. 낮을수록 mask 넓어짐
PIL_WEIGHT_NORMALIZE  = True     # True: scale PIL weight to [0, 1]; False: keep raw Gaussian product
WANG_PIL_EPS          = 30       # Wang et al. PIL core neighborhood radius
WANG_PIL_PORTION      = 0.125    # Wang et al. core density portion
WANG_PIL_SIGMA        = 10.0     # Gaussian sigma used after core extraction
WANG_PIL_KERNEL_RADIUS = 31      # 63x63 Gaussian kernel
WANG_PIL_MIN_STRENGTH = 0.0      # 0 keeps the original Wang-style product mask

# ───────── 5. 라벨링 ─────────
PRED_WINDOW_HOURS  = 24
FLARE_THRESHOLD    = "M1.0"

# ───────── 6. 분할 ─────────
SPLIT_RATIO = (0.7, 0.1, 0.2)
SPLIT_METHOD = "ar_month_stratified"
SPLIT_STRATIFIED_TRIALS = 512
SPLIT_PURGE_HOURS = 24
SPLIT_MIN_VALID_FRACTION = 0.10
RANDOM_SEED = 42

# ───────── 7. 학습 ─────────
BATCH_SIZE      = 64
NUM_WORKERS     = 4
EPOCHS          = 100
'''TRAIN_EPOCH_NOTIFY_EVERY = 1
'''
EARLY_STOP_PATIENCE = 30
EARLY_STOP_MIN_EPOCHS = 10
'''
EARLY_STOP_MIN_DELTA = 0.001
TRAIN_PAUSE_CHECK_SECONDS = 30
LR              = 4e-5
WEIGHT_DECAY    = 5e-2
LR_SCHEDULER    = "batch_step"
LR_STEP_BATCHES = 320
LR_STEP_FACTOR  = 0.5
ADAM_BETA1      = 0.9
ADAM_BETA2      = 0.95
DEVICE          = "directml"
'''
DENSENET_VARIANT = "densenet121"
'''
TRAIN_VARIANT = "full"
TRAIN_DROPOUT = 0.5
TRAIN_LOSS = "flare"            # fixed: Takagi-style binary FLARE loss
TRAIN_POS_WEIGHT_SCALE = 1.0
TRAIN_INPUT_CLIP = 1.0           # max-abs normalization keeps inputs in [-1, 1]
FLARE_LAMBDA_BSS = 3.0
FLARE_IB_WARMUP_EPOCHS = 2
MASK_SHORTCUT_AUG = "off"         # fixed: enable for masked and full training
MASK_BG_NOISE_PROB = 0.8         # randomize zeroed background during masked training
MASK_BG_NOISE_STD = 0.04         # normalized units; effective when normalize=True
MASK_EDGE_DROP_PROB = 0.8        # randomize the hard mask boundary band
MASK_EDGE_ERODE_MAX = 3          # max pixels used to define boundary jitter band
MASK_SHIFT_PX = 24               # stronger random translation to reduce position/padding shortcuts
TRAIN_RECROP_PROB = 0.8          # random crop-and-pad probability
TRAIN_RECROP_PX = 12             # maximum pixels cropped from each edge
TRAIN_INTENSITY_JITTER_PROB = 0.5
TRAIN_INTENSITY_JITTER = 0.05    # per-channel multiplicative jitter (+/- fraction)
HARP_BALANCED_SAMPLER = "on"     # equal number of train frames from every HARP per epoch
HARP_FRAMES_PER_EPOCH = 8        # frames sampled from each HARP per training epoch
HARP_REFERENCE_FRAMES = 4        # fixed frames per HARP used for comparable train AUC
MAX_TRAIN_VALID_AUC_GAP = 0.10   # train-reference vs valid AUC warning threshold only
OPEN_TRAIN_CURVES = False        # run_pipeline passes --open-curves for masked/full train steps
GAP_ALERT_EVERY = 1              # send ntfy warning every N epochs when the AUC gap exceeds threshold
LR_PLATEAU_FACTOR = 0.5          # ReduceLROnPlateau multiplicative decay
LR_PLATEAU_PATIENCE = 2          # epochs without valid AUC improvement before decay
LR_MIN = 1e-6                    # lower LR bound'''

# ───────── 후처리 병렬화 ─────────
PIL_WORKERS   = 6
PIL_BATCH_SIZE = 5000
PIL_CHUNKSIZE = 4
PIL_LOCAL_RAW_DIR = RAW_DIR
PIL_DELETE_LOCAL_RAW_AFTER_USE = True

# ───────── 8. I/O 최적화 ─────────
PREFETCH_FACTOR = 16
HDF5_RDCC_NBYTES = 256 * 1024 * 1024
HDF5_RDCC_NSLOTS = 100003
HDF5_RDCC_W0 = 0.75
HDF5_COMPRESSION = "lzf"
HDF5_GZIP_LEVEL = 1
SAVE_LAST_EVERY = 5
