# Benchlab Solar Wind Prediction — Docker 제출 작업 브리프

> Claude Code 세션 컨텍스트용. 최종 갱신 2026-11-08.
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

## 6. 모델 코드 분석 (`syzygy_code/`, 브랜치 `wooseong-art-Syzygy_code`에서 가져옴)

**구성**: 추론 진입점 `fc_predict.py`(`predict(t0, {"mode":"live"})`), 모델 정의 `fc_model.py`(RegimeNetCF), 데이터 수집·전처리 `sw_utils.py`, 피처 `build_features.py`/`04b_ch_features.py`, PFSS `04f`/`04m`/`pfss.py`/`pfss_trace.py`, `common.py`. **학습 코드는 없음**(추론에 불필요).

**앙상블**: G01(4×3 격자, 기본·fallback), P2o, O(HSS 점유), T2(onset), RegimeNetCF(영상+ACE), Router(MoE, masked AIA+PFSS). 최종 = `final_selection.json`의 base(A/B/C/D) + 보정. **CME(DONKI) 미사용**.

**입력**
- ACE 시간별 V, [t0 − 29일, t0). 파생: hist 48h, persist 24h, v_now(6h 평균), rec(t0−28d~−22d 6h bin 24개), v27, recur. 정규화 `(V − 430) / 100`, 결측 0 + mask.
- AIA 193 + HMI, slot = t0 − 12k h (k = 0…10), 없으면 slot − 6h. RegimeNet 0/24/48/72/96h × 2채널(128²), CH 특징 0–96h 12h 간격, 격자 G 0/24/48/72h, Router masked AIA(t0).
- PFSS(`router_variant == "fs"`일 때): 최근 28일 HMI 6h 간격(최대 약 112장) → 15분 내 가능 여부 실측 필요. `br` 변형은 live 미구현 → fallback.

**출력 불일치**: 모델은 t0를 6h 격자로 내린 뒤 **h = 24/48/72/96의 6h 평균 4개**(+ RegimeNet/Router 16 bin)를 냄 → **시간별 72개로 변환 규칙 필요**.

**실행 불가 요인 (팀원에게 요청 중)**
1. `config.py`가 **플레어 프로젝트 파일**(`E:\flare_data`, SHARP). 코드가 쓰는 `C.*` 117개 중 113개 없음. 원본은 `D:\syzygy\code\config.py`. 정규화·전처리·모델 구조 값은 추측 금지.
2. 빠진 모듈: `model.py`, `train.py`, `run_phase.py`, `cr_train.py`, `fc_common.py`, `oc_common.py`, `oc_predict.py`, `oc_calibrate.py`, `oc_onset.py`, `monitoring/logger`.
3. 체크포인트(seed별 G01/P2o/O/T2/RegimeNetCF/Router), `final_selection.json`, `ensemble_meta_valid.json`, `preds_info.json`, `boxcox_*.pkl`, `ch_pfss.h5` `feat_fs` 정규화 attrs, `IMAGE_INDEX_CSV`.

**`sw_utils.py`를 Docker에 쓸 때 고칠 점**: `http_get`이 예외를 던지고 호출부가 안 잡음(→ exit 1 위험) / 공개 지연 없음(`not_after = T`) / 확정본 우선이라 과거 t0 테스트에서 누수 / 재현성 로그 없음 / timeout 300초, 순차 다운로드 / 실시간에서 L2 요청 낭비 / config import 시 Windows 경로 `mkdir`, 캐시를 코드 폴더에 씀(→ `/tmp`, `monitor=False`).

## 7. 현재 구현 (`submission_pack/`, 단계 B 완료)

```
run.sh <t0>  →  python3 /app/predict.py "$T0" "$OUT" || true; /app/emergency.sh "$T0" "$OUT"  →  upload.sh
predict.py
 ├─ ⓪ 기후값 430 × 72 즉시 저장
 ├─ watchdog (SYZYGY_BUDGET_S, 기본 600초) → os._exit(0)
 ├─ ① ace.load_ace_hourly(t0) (예산 180초) → baseline.forecast → 저장
 ├─ ② model_stage.predict(t0, ace_hourly, deadline) → 저장   ← 단계 C에서 구현 (MODEL_AVAILABLE=False)
 └─ 각 저장은 원자적(os.replace). sanitize: 72개 아니면 폐기, NaN→430, clip 200–1200
```

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
- `requirements.txt`는 현재 비어 있음(주석만). `submission.json`: Syzygy / min070118@gmail.com / 4096 / 16384 / 도메인 4개(sohoftp, cdaweb, services.swpc, jsoc1).
- 예제 `model.py` 삭제. 제출 ZIP: `Syzygy/submission_B.zip` (`.gitignore`에 `submission_*.zip`).

**검증 결과 (2026-11-08)**
- `test_local.sh` PASS: 템플릿 원본(1606 MiB) / 우리 버전(197 MiB, 실행 9초).
- ACE 수집 약 6–8초, 파일 30개, 커버리지 99.6%. 과거 t0(11-01 06:00) 마지막 사용 시간 05:00 → 누수 없음.
- Baseline 백테스트(2026-06~11, t0 512개, ACE RT 정답): MAE persistence 70.2 / 재귀 64.1 / τ=24h 52.2 / **τ=48h 51.0** / τ=96h 55.1 km/s.
- 장애 테스트 모두 exit 0 + 72개: `--network none`(기후값), Python 없음(emergency), 예산 2초(watchdog), 잘못된 t0.
- SDO: 실시간·과거 t0 모두 미공개 파일 제외 확인. **HMI는 t0 slot에서 못 찾음**(nrt 지연 약 50분 > 허용 ±60분 경계) → 단계 C에서 허용 오차 결정.

## 8. 테스트 방법

- 로컬 테스트: `cd submission_pack/local_tests && bash test_local.sh <t0>`.
  - **Windows 주의**: `python3`가 Microsoft Store 바로가기라 mock 서버가 안 뜸 → `exec python "$@"`인 `python3` shim을 만들어 PATH 앞에 둘 것. PATH에는 `/c/...` 형식 사용(`C:/...`는 콜론 때문에 깨짐).
- 장애 테스트: `docker run --rm --network none --entrypoint bash swp-local -c "python3 /app/predict.py <t0> /tmp/f.json; …"`, `-e SYZYGY_BUDGET_S=2`, `PATH=/usr/bin:/bin`(Python 없음).
- 과거 t0 여러 개로 컷오프 누수 확인. HMI nrt는 4–5일 뒤 삭제되므로 오래된 t0는 영상 재현 불가.

## 9. 남은 작업

**단계 B 마무리**
- [ ] B-7 실제 제출로 `ELIGIBLE` 확보 — **사용자 승인 필요**. 기존 `submit.py`는 conformance 전용(`conformance_pack` + 해시 검사) → `submission_pack`용 복사본 필요. PORTAL은 `.team`.
- [ ] 변경사항 커밋 (아직 미커밋)

**단계 C (원본 config·모듈·가중치 수령 후)**
1. 원본 config로 입력 사양 확정 (변수, 창 길이, 정규화, 전처리, `IMG_MATCH_TOL_MIN`)
2. `sw_utils` 보완 또는 `syzygy_live`로 대체: LAG, 예외 처리, 실시간 L2 생략, 병렬 다운로드
3. `fc_predict` → `model_stage.predict` 연결: 경로 `/tmp`, `monitor=False`, Windows 경로 제거
4. **6h bin → 시간별 72개 변환 규칙** 결정
5. PFSS 사용 여부(`router_variant`), 15분 내 실행 실측
6. 과거 t0 백테스트: 모델 vs baseline MAE
7. `requirements.txt` 학습 버전으로 고정, 이미지 크기·실행 시간 측정, CPU/메모리 확정
8. README 갱신, 최종 제출 (재제출 여유 확보)

## 10. 미확정 / 확인 필요

- 원본 `config.py`, 빠진 모듈, 체크포인트 (팀원 요청 중).
- `forecast[k]`와 ACE 시간 라벨의 정확한 대응(t0+(k+1)h가 시간 시작인지 끝인지) — 주최측 확인.
- 품질 플래그 처리 방식, live 시작일 (주최측 공지 대기).
- 최종 Participation Terms 전문 (6.2–6.8절).
- ~~conformance 완료 여부~~ → **완료** (커밋 0534dfd). ~~팀명~~ → `Syzygy` (`.team`).
