# Benchlab Solar Wind Prediction — Docker 제출 작업 브리프

> Claude Code 세션 컨텍스트용. 최종 갱신 2026-11-10.
> 출처: starter-kit 레포(`wooseong-art/Syzygy`, branch `starter-kit`), Slack `#benchlab-support-and-q`, 주최측 공지 PDF(2026-11-03), 엔드포인트 실측(2026-11-08).

## 0. 목표와 마감

- 이미 학습된 태양풍 속도 예측 모델을 **Benchlab 제출용 Docker 이미지**로 패키징한다.
- 트랙: **Operational Podium (100% in-container)** — 상금($20k) 대상. 외부 연산 금지.
- **코드 프리즈: 2026-11-19 23:59:59 UTC (= KST 11-20 08:59)**. 그 전에 실제 제출이 dry run을 통과해 `ELIGIBLE` 상태여야 함.
- 재제출하면 이전 ELIGIBLE 버전이 대체됨 → 새 버전이 실패하면 자격을 잃을 수 있음. 마감 직전 무리한 변경 금지.

## 1. 실행 계약 (플랫폼이 강제)

| 항목 | 내용 |
|---|---|
| 실행 | `<ENTRYPOINT> <t0>` — t0는 argv[1], 예: `20261111T060000Z`. `$INFERENCE_TIMESTAMP`로도 제공 |
| 주기 | 6시간마다 **새 컨테이너**로 실행 (stateless, 캐시/이전 파일 없음) |
| 출력 | t0+1h … t0+72h, **정확히 72개** 시간별 속도(km/s). `forecast[0]` = t0+1h |
| 출력 형식 | JSON, ≤ 1 MiB. 플랫폼은 `forecast_speed_kms`만 읽음 (다른 키는 자유) |
| 업로드 | `$OUTPUT_POST`(presigned S3 POST JSON)로 `upload.sh`가 전송 |
| 시간 제한 | **15분, 이미지 pull 포함** (pull ≈ 1 GB당 10초 추정) |
| 자원 | `submission.json` 선언값. 최대 8 vCPU / 32 GB / 디스크 ~20 GB, **CPU only (GPU 없음)** |
| 네트워크 | 공개 인터넷 사용 가능, 실행 1회당 20 GB fair-use |
| 아키텍처 | `linux/amd64` |
| 이미지 크기 | < 15 GiB (작을수록 좋음) |
| 사용자 | non-root `runner`(uid 1000). 쓰기는 `/tmp` 사용 |
| 성공 조건 | **exit 0 AND 유효한 forecast 업로드** — 둘 다 충족해야 함 |

- 로컬 테스트는 값 범위 0–10000, NaN/null 없음을 검사함. 반드시 유한한 float 72개를 출력할 것.

## 2. 제출 패키지 규칙

- ZIP **루트**에 `Dockerfile`, `submission.json`, `README.md` 필수. 폴더째가 아니라 내용물을 압축.
- Dockerfile은 EDIT 마커 사이만 수정. 상단 PLATFORM LAYER(`ca-certificates curl jq`, `upload.sh`)와 하단 `useradd runner` / `USER runner` / `ENTRYPOINT ["/app/run.sh"]` 유지.
- `run.sh`는 실행 한 줄만 수정. `upload.sh`는 수정 금지.
- 유효한 Fargate 조합만 허용: 1024 → 2048–8192, 2048 → 4096–16384, 4096 → 8192–30720, 8192 → 16384–32768 (MiB, 1024 단위, 8 vCPU는 4096 단위).
- `declared_domains`: **실행 중 접속하는 모든 호스트** (리다이렉트 대상 포함). 빌드 때만 쓰는 pypi 등은 제외.
- 베이스 이미지는 ECR 미러(`public.ecr.aws/docker/library/python:3.11-slim`). PyTorch는 CPU 휠(`--extra-index-url https://download.pytorch.org/whl/cpu`, `torch==X+cpu`).

## 3. 대회 규칙 — 데이터/연산 (11-03 공지)

**허용**: 공개 관측 데이터와 공개 모델 산출물(SDO/HMI, GONG, ADAPT, OMNIWeb, CDAWeb, NOAA SWPC 등). 학습은 외부(GPU)에서 해도 됨.

**금지 (Operational Podium)**
- 우리 팀 전용으로 컨테이너 밖에서 수행한 연산, 피처 생성, 추론 (공개 URL에 올려도 동일).
- **Pass-through 금지**: 외부 예보를 그대로 넘기거나 단순 보간만 하는 것.
- 라이브 평가 중 재학습, 코드 수정, 수동 개입.

**t0 정보 컷오프 (가장 중요)**
- t0 시점까지 **공개(publish)된** 정보만 사용. "The fetch may happen after t0; the information may not."
- 관측 timestamp가 t0 이전이라는 것만으로는 부족. 공개 시각을 모르면 보수적 지연(LAG)을 적용하고 README에 문서화.
- 실시간 API와 아카이브는 t0 이후 데이터도 반환하므로 **직접 잘라낼 것**.

**재현성 (권장)**: 다운로드마다 URL, 받은 시각, Last-Modified, sha256을 stdout 로그로. 주최측 재실행 시 원본 변경 증거가 되면 실패로 보지 않음.

## 4. 채점

- 정답: **ACE/SWEPAM 1시간 Bulk Speed**. 실시간 리더보드는 SWPC `ace_swepam_1h.json`.
- 결측과 fill value는 제외. 품질 플래그 처리는 추후 공지.
- **예측값이 빠진 시간은 persistence(t0 이전 마지막 유효 ACE 값)로 대체되어 채점됨.**

## 5. 데이터 소스 (2026-11-08 실측 확정)

| 소스 | URL | 간격 / 공개 지연 | 비고 |
|---|---|---|---|
| **ACE RT 1분 일별** (주 소스) | `https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/YYYYMMDD_ace_swepam_1m.txt` | 1분 / 수 분. 당일 파일 계속 갱신, 지난 날짜는 다음날 00:11 확정 | 행 `YYYY MM DD HHMM MJD SOD S Np V T`, **S=0만 유효**(9=결측, -9999.9). 최소 수개월치 보관 → 29일 창 전체 커버 |
| ACE K0 (HAPI) | `https://cdaweb.gsfc.nasa.gov/hapi/data?id=AC_K0_SWE&parameters=Vp&time.min=…Z&time.max=…Z` | 5분 / **약 7–8일** | 데이터 없으면 HTTP 200 + `{"status":{"code":1201}}`. fill `-1.0E31` |
| ACE H2 (HAPI) | `id=AC_H2_SWE` | 1시간 / **2024-07-09에서 중단** | 학습용만 |
| CDAS REST | `https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/{ID}/data/{시작Z},{끝Z}/Vp?format=text` | — | 2단계: `Accept: application/json` → `FileDescription[].Name` txt 재요청. 6개월 초과 시 503 |
| RTSW 1분 | `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | 1분 / 1–2분, **최근 24시간만** | ACE·SOLAR1·IMAP 혼재(`active`=SOLAR1). `source=="ACE"`, `overall_quality==0`로 필터 |
| SWPC ACE 1h | `https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json` | 1시간 / 약 30일치 | 리더보드 기준. `plasma-7-day.json`, `rtsw_wind_1m_7_day.json`은 **404** |
| AIA 193 NRT | `https://jsoc1.stanford.edu/data/aia/synoptic/nrt/YYYY/MM/DD/H{HH}00/AIA{YYYYMMDD}_{HHMMSS}_0193.fits` | 3분 / 15–20분 | 파일명 **HHMMSS**. 빈 구간 안 채워짐 → 디렉터리 목록에서 선택. 약 1.2 MB |
| AIA 193 확정본 | 위 경로에서 `/nrt` 제외 | 2분 / **약 7일** | 실시간 불가 |
| HMI 1k | `https://jsoc1.stanford.edu/data/hmi/fits/YYYY/MM/DD/hmi.M_720s[_nrt].YYYYMMDD_HH0000_TAI.fits` | 1시간 / nrt 약 50분, 확정본 4–5일 | **확정본이 생기면 nrt 삭제**. 시각은 TAI(UTC+37초). 약 1 MB |

- JSOC 디렉터리 목록에 파일별 수정 시각(분 단위, UTC)이 표시됨 → 공개 시각으로 사용.
- 최신 시각 확인: `…/synoptic/nrt/image_times`, `…/hmi/fits/latest_fits_time`.

## 6. 모델 코드 (`wooseong-art/git_repo`, main, 2026-11-10 분석)

**출처**: `https://github.com/wooseong-art/git_repo` (커밋 1333bb2, 447 파일, 80 MB). 이전 `syzygy_code/`(브랜치 `wooseong-art-Syzygy_code`)는 config가 플레어 프로젝트 파일이고 모듈·가중치가 빠져 있었음 → **이 레포로 대체**. 추론 진입점 `code/experiments/syzygy_final_cme_free_2026/fc_predict.py`(`predict(t0, {"mode":"live", ...})`).

**받은 것 (이전 "실행 불가 요인" 모두 해소)**
- 원본 `code/config.py`(태양풍용, `BASE_DIR` = `SW_BASE_DIR` 또는 레포 루트). import closure 정적 확인 → 빠진 로컬 모듈 없음. 외부 `proswin`/`hss_prediction`은 lazy import라 추론에 불필요.
- 가중치 30개(G01·RegimeNetCF·RouterR2·O ×5 seed, 출력만 하는 P2o·T2 ×5): 원래 위치 + `final_bundle/weights/`. 설정 `final_selection.json`, `ensemble_meta_valid.json`, `preds_info.json`, `model_preds_info.json`, `boxcox_maps_A/C.pkl`.
- `ch_pfss.h5` `feat_fs` 상수 → `final_bundle/final_params.py`·`configs/pipeline_params.json` (h5 없으면 코드가 자동 사용). `IMAGE_INDEX_CSV` 없으면 live 캐시만 사용(이미 수정됨).
- `requirements.txt`(추론용, 개발 Python **3.12.3**): torch 2.4.1(+cpu), numpy 2.2.0, pandas 2.2.3, h5py 3.16.0, scipy 1.15.2, astropy 7.1.0, sunpy 7.1.2, requests 2.32.5, tqdm 4.67.1.

**최종 모델 `C+HBC0.25`** (`final_bundle/README.md`)
```
v       = 0.5·mean_seed(G01) + 0.5·mean_seed(Router R2)
Router_s = denorm(anchor_s + Σ softmax(router_s)·ΔV_s)   (anchor·ΔV = 같은 seed RegimeNetCF)
v_final = v + 0.25·p_occ·(BoxCox_C(v) − v),  p_occ = mean_seed(sigmoid(O occ_logits))[bins 3,7,11,15]
fallback: 영상·PFSS 입력 하나라도 없으면 v = G01, 보정 boxcox_maps_A
```
- 성능(RMSE km/s, 4 horizon): 최종 valid 63.98 / test(탐색) 65.57, G01 67.69 / 67.00. test 이득 −1.4(CI 0 포함). CME(DONKI) 미사용.

**입력**
- ACE 시간별 V, [t0 − 696h, t0). 정규화 `(V − 430) / 100`, 결측 0 + mask.
- AIA 193 + HMI, slot = t0 − 12k h (k = 0…10, 22장). slot 이전 **3시간**(`IMG_MATCH_TOL_MIN=180`) 안 가장 가까운 정상 영상, 없으면 slot − 6h(`IMG_FILL_TOL_HOURS`). 전처리 256² → 128². G01은 격자 G(0/24/48/72h), RegimeNet 0–96h 24h 간격, CH 특징 12h 간격, Router masked AIA(t0).
- PFSS f_s(R2): [T − 28일, T − 2h] HMI 6h 간격 → Carrington 지도 → l ≤ 20 → CH 자기력선 추적 → f_s 5분위.
- 인과성: 영상 obs ≤ slot, PFSS HMI obs ≤ T − 2h, ACE hour < t0. **공개 시각은 보지 않음** + 확정본 우선 → 실시간은 괜찮지만 과거 t0 백테스트는 누수.

**출력 불일치**: t0를 6h 격자로 내린 뒤 **h = 24/48/72/96의 ACE 6h 평균 [t0+h−6, t0+h) 4개**(Router/RegimeNet은 16 bin) → **시간별 72개 변환 규칙 필요**.

**팀원 측 검증**: archive ↔ live 입력 과거 4시각 비트 일치, 예보 차 ≤ 0.01 km/s, 모델 계산 0.07초, live 1회(2026-11-04 12:00) 상태 OK. **단, 캐시 빈 상태에서 입력 준비 5,466초**(한국에서 JSOC 103건 순차, 101장).

**Docker에 쓸 때 고칠 점**
1. **영상 다운로드 시간 (최대 위험)**: stateless라 매 실행 FITS 약 130장(영상 22 + PFSS HMI 약 107, 약 150 MB) + 디렉터리 목록 수십 건. 병렬화 + 예산 초과 시 PFSS 생략 → G01 fallback 필수. AWS에서 실측 필요.
   - 실측(2026-11-10, 한국 PC → JSOC): 디렉터리 목록 0.4–1.0초, HMI 1.04 MB 순차 1.5–3.7초/장(약 0.3 MB/s), **8 병렬 24장 17.2초**(합계 약 1.4 MB/s), AIA NRT 1.2 MB 9.2초(1장). → 130장 추정 순차 약 7–10분 / 8 병렬 약 1.5–2분. 5,466초는 다운로드 자체보다 재시도·timeout·순차 목록 탐색 영향으로 추정.
2. ACE: `sw_utils.load_ace_hourly`는 L2 HAPI → L2 CDAS → K0 순, timeout 300초(L2는 2024 중단) → `syzygy_live.ace` 결과 주입(`live_inputs`의 `ace_source` 분기 활용).
3. `http_get`이 재시도 후 예외 → `predict()` 전체 중단. 별도 프로세스/try로 격리.
4. 쓰기: `LIVE_CACHE` = 코드 폴더 `features/live_images`, `monitor=True` 기본(monitoring/, predictions/) → `/tmp`로, `monitor=False`.
5. 과거 t0 테스트 누수 → `syzygy_live.sdo`(목록 mtime 기준)로 후보 필터.
6. astropy IERS 자동 다운로드 가능성(미선언 도메인) → 컨테이너에서 확인, 필요 시 `auto_download=False`.
7. 베이스 이미지 Python 3.12-slim으로 맞추는 것 검토(개발 3.12.3).
8. 실시간 ACE(RT) vs 학습(L2/K0) 차이 bias +2.4 km/s, G01 RMSE +0.7–1.2 (감수 가능).

## 7. 현재 구현 (`submission_pack/`, 단계 B 완료 + 단계 C 모델 연결 2026-11-10)

```
run.sh <t0>  →  python3 /app/predict.py "$T0" "$OUT" || true; /app/emergency.sh "$T0" "$OUT"  →  upload.sh
predict.py
 ├─ ⓪ 기후값 430 × 72 즉시 저장
 ├─ watchdog (SYZYGY_BUDGET_S, 기본 600초) → os._exit(0)
 ├─ ① ace.load_ace_hourly(t0) (예산 180초) → baseline.forecast → 저장
 ├─ ② model_stage.predict(t0, ace_hourly, deadline, ace_source) → 자식 프로세스 `python3 -m syzygy_model.runner`
 │     → (72개 | None, info). 72개면 stage "model"로 저장, None이면 baseline 유지 + info.model에 모델 결과 기록
 └─ 각 저장은 원자적(os.replace). sanitize: 72개 아니면 폐기, NaN→430, clip 200–1200
```

**모델 연결 구조 (팀원 코드는 수정하지 않음)**
- `model/`: `tools/sync_model.py <git_repo 경로>`가 git_repo에서 **그대로 복사**(같은 상대 경로). import closure + `cr_events`가 import 시 부르는 `external/proswin` 6개 모듈 + Pipeline이 읽는 json/pkl/pt. `model/MANIFEST.json`에 commit·sha256. **모델 코드 변경 시: 스크립트 재실행 → 변경 파일만 표시 → 재빌드.** `tools/`는 이미지에서 제외. **git에는 `model/MANIFEST.json`만 추적**(`.gitignore`: `submission_pack/model/*`, `!…/MANIFEST.json`) → 새로 클론하면 sync 스크립트로 `model/` 생성 후 빌드.
- `syzygy_model/runner.py`: 실행 중 패치만 함 — `LIVE_CACHE` → `/tmp/syzygy_cache`, `sw_utils.http_get` → 예외 없음·짧은 timeout·재현성 로그·선다운로드 캐시, `sw_utils.load_ace_hourly` → syzygy_live ACE 주입, `monitor=False`, astropy IERS 자동 다운로드 끔. 영상 slot 22개 + PFSS HMI(최신부터) **병렬 선다운로드**(`SYZYGY_DL_WORKERS`, 기본 12) 동안 Pipeline 로드. 네트워크 마감 = 모델 예산 − `SYZYGY_COMPUTE_RESERVE_S`(150초).
- `syzygy_model/hourly.py`: **변환 규칙 미정 → `to_hourly()`가 None** (규칙 확정 후 구현). 가드: `ACE_48h` 결측이면 None.
- 이미지: Python **3.12-slim**, requirements = 팀원 버전 고정(torch 2.4.1+cpu 등).

| 파일 | 역할 |
|---|---|
| `predict.py` | 전체 흐름, 항상 exit 0. 로그는 stdout JSON, 예측은 파일. 출력에 `stage`, `info` 포함 |
| `syzygy_live/net.py` | `get()` timeout·재시도·전역 예산, 예외 안 던짐. url/sha256/Last-Modified 로그 |
| `syzygy_live/ace.py` | RT 일별(8 병렬) → K0(24h 이전 공백 ≥ 6h) → RTSW(최근 24h 공백). 컷오프 obs ≤ t0 − 5분. 시간 평균 최소 RT 10개 / K0 3개 |
| `syzygy_live/sdo.py` | `find_aia` / `find_hmi` / `download`. obs ∈ [slot − 60분, min(slot, t0)], 목록 mtime + 1분 ≤ t0. 확정본 우선(공개 조건으로 누수 자동 차단), HMI TAI→UTC |
| `syzygy_live/baseline.py` | `w·v_now + (1−w)·v_rec`, w = exp(−k/48), v_rec = 655h 전 ±3h 평균 |
| `model_stage.py` | 모델 자리(인터페이스 명시) |
| `emergency.sh` | 유효 파일이 없을 때만 430 × 72 작성 |

- `syzygy_live/`는 **표준 라이브러리만** 사용 → 모델 스택이 깨져도 baseline 동작.
- `requirements.txt`: 팀원 개발 버전 고정(2026-11-10). `submission.json`: Syzygy / min070118@gmail.com / 4096 / 16384 / 도메인 4개(sohoftp, cdaweb, services.swpc, jsoc1).
- 예제 `model.py` 삭제. 제출 ZIP: `Syzygy/submission_B.zip` (`.gitignore`에 `submission_*.zip`).

**검증 결과 (2026-11-08)**
- `test_local.sh` PASS: 템플릿 원본(1606 MiB) / 우리 버전(197 MiB, 실행 9초).
- ACE 수집 약 6–8초, 파일 30개, 커버리지 99.6%. 과거 t0(11-01 06:00) 마지막 사용 시간 05:00 → 누수 없음.
- Baseline 백테스트(2026-06~11, t0 512개, ACE RT 정답): MAE persistence 70.2 / 재귀 64.1 / τ=24h 52.2 / **τ=48h 51.0** / τ=96h 55.1 km/s.
- 장애 테스트 모두 exit 0 + 72개: `--network none`(기후값), Python 없음(emergency), 예산 2초(watchdog), 잘못된 t0.
- SDO: 실시간·과거 t0 모두 미공개 파일 제외 확인. **HMI는 t0 slot에서 못 찾음**(nrt 지연 약 50분 > 허용 ±60분 경계) → 단계 C에서 허용 오차 결정.

**검증 결과 — 모델 연결 (2026-11-10, 한국 PC, t0 = 2026-11-10 06:00)**
- 이미지 **1,949 MiB**(pull 약 20초). `test_local.sh` **PASS, runtime 161초**. 컨테이너 메모리 최대 **1.08 GB**.
- 모델 state **OK**, 결측 입력 없음, `C+HBC0.25` = 627.8 / 556.8 / 492.1 / 462.3 km/s (h = 24/48/72/96). 실행마다 동일.
- 선다운로드 FITS 123장(128 MB), 영상 22/22, PFSS HMI 101/102(1 slot JSOC에 없음). 시간: 8 workers 90초 / 12 workers 108초 / 16 workers 57초(JSOC 상태 따라 변동). 선다운로드 후 추가 요청 0건. predict(전처리 + PFSS + 30개 망) 24–35초, Pipeline 로드 0.4초.
- 접속 호스트: jsoc1, sohoftp(필요 시 cdaweb, services.swpc) → 선언 4개 안. astropy/sunpy 다운로드 흔적 없음(`/home/runner`에 캐시 파일 없음). 오류는 확정본 AIA 폴더 404(정상)뿐.
- 장애: `--network none` → exit 0, 72개(baseline 단계), 모델은 FALLBACK_G01(ACE_48h·영상 결측). `SYZYGY_BUDGET_S=200` → 영상 일부만 받고 FALLBACK_G01, exit 0.

## 8. 테스트 방법

- 로컬 테스트: `cd submission_pack/local_tests && bash test_local.sh <t0>`.
  - **Windows 주의**: `python3`가 Microsoft Store 바로가기라 mock 서버가 안 뜸 → `exec python "$@"`인 `python3` shim을 만들어 PATH 앞에 둘 것. PATH에는 `/c/...` 형식 사용(`C:/...`는 콜론 때문에 깨짐).
- 장애 테스트: `docker run --rm --network none --entrypoint bash swp-local -c "python3 /app/predict.py <t0> /tmp/f.json; …"`, `-e SYZYGY_BUDGET_S=2`, `PATH=/usr/bin:/bin`(Python 없음).
- 과거 t0 여러 개로 컷오프 누수 확인. HMI nrt는 4–5일 뒤 삭제되므로 오래된 t0는 영상 재현 불가.

## 9. 남은 작업

(2026-11-10 기준. 마감 2026-11-19 23:59:59 UTC까지 9일)

**완료**
- ~~원본 config·모듈·가중치 수령~~ (`git_repo`, §6) / ~~모델 연결~~ (`syzygy_model/runner.py`, 팀원 코드 무수정) / ~~병렬 선다운로드·ACE 주입·예외 격리·`/tmp`~~ / ~~requirements 고정, Python 3.12-slim~~ / ~~로컬 PASS 161초, 장애 테스트~~ (§7)
- ~~`model/` gitignore + MANIFEST만 추적~~ (커밋 5f75b00, `docker` 브랜치)

**보류 (사용자 지시)**
- [ ] **B-7 실제 제출로 `ELIGIBLE` 확보** — 나중에 진행. **사용자 승인 필요**. `submit.py`는 conformance 전용 → `submission_pack`용 복사본 필요. PORTAL은 `.team`.
- [ ] **시간별 72개 변환 규칙** (`syzygy_model/hourly.py`) — 규칙 확정 후 진행. 확정 전에는 예보 = baseline, 모델 결과는 `info.model`에만 기록.
- [ ] **모델 코드 변경분 반영** — 팀원 수정본 수령 시 `python tools/sync_model.py <git_repo>` → 변경 파일 확인 → 재빌드 → `test_local.sh`.

**진행 가능 / 다음 작업**
1. [ ] 나머지 변경사항 커밋 (Dockerfile, predict.py, model_stage.py, requirements.txt, net.py, syzygy_model/, tools/, CLAUDE.md) — 사용자 확인 후
2. [ ] 과거 t0 테스트용 **영상 공개 시각 필터**(JSOC 목록 mtime ≤ t0) runner에 추가 — 백테스트 전 필수 (실시간 운영에는 영향 없음)
3. [ ] 변환 규칙 백테스트 준비: 팀원 archive 예측(과거 t0별 `g01_kms`, `router_cf_16bin_kms`, `final_speed_kms`, `v_now_kms`) 수령 → ACE 시간별 정답으로 후보 규칙 MAE 비교 (보간 / baseline 혼합 / Router 16 bin 모양)
4. [ ] 모델 vs baseline 백테스트 (변환 규칙 적용 후)
5. [ ] 결정 대기 항목 반영 (§10): PFSS 부분 다운로드 처리, 자원 선언
6. [ ] README 갱신 (데이터 소스, 공개 지연 LAG, 변환 규칙, fallback 단계)
7. [ ] 최종 제출 — 재제출 실패 위험 고려해 마감 2–3일 전, 제출 ZIP은 따로 보관 (`model/` 포함)

## 10. 미확정 / 확인 필요

- ~~원본 `config.py`, 빠진 모듈, 체크포인트~~ → `git_repo`로 수령 (2026-11-10).
- 팀원에게 추가 요청: ① 11-04 12:00 live 실행의 `features/live_images/*.npy`·`hmi_index.csv`·`forecast_20261104T1200_live.json`·`reports/live_input_check.json`·`inference_check.json` (컨테이너 결과 비트 대조용) ② **archive 예측**(과거 t0별 g01·router 16 bin·final·v_now, 변환 규칙 백테스트용) ③ 시간별 72개 변환 규칙 합의 ④ 모델 코드 수정 예정분 ⑤ (선택) Router **R0** 가중치 — 로컬 실측상 PFSS 시간 문제 작아 우선순위 낮음.
- 사용자 결정 대기: ① Python **3.12-slim** 사용 확인 (§2 템플릿은 3.11, 팀원 개발 3.12.3) ② PFSS용 HMI 부분 다운로드 시 그대로 계산(현재) vs 일정 비율 미만이면 G01 fallback ③ 자원 4 vCPU / 16 GB 유지(권장, 실측 최대 1.08 GB) vs 축소.
- `forecast[k]`와 ACE 시간 라벨의 정확한 대응(t0+(k+1)h가 시간 시작인지 끝인지) — 주최측 확인.
- 품질 플래그 처리 방식, live 시작일 (주최측 공지 대기).
- 최종 Participation Terms 전문 (6.2–6.8절).
- ~~conformance 완료 여부~~ → **완료** (커밋 0534dfd). ~~팀명~~ → `Syzygy` (`.team`).
