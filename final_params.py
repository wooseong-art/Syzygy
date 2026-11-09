"""
final_params.py — SYZYGY FINAL (CME-free) 최종 모델 C+HBC0.25 추론 값 (JSON·pkl 대신 쓰는 상수 모음).

자동 생성: make_final_params.py, 2026-11-09T03:46:04Z. 손으로 고치지 말 것.
원본 (값을 그대로 읽어 옮김):
  final_selection.json (조합·보정 선택), boxcox_maps_{C,A}.pkl (BoxCoxDistributionMap.params()),
  D:/syzygy/ch_pfss.h5 feat_fs 속성 (f_s 표준화), model_preds_info.json (T2 τ), final_bundle/MANIFEST.json (sha256),
  config.py / physical_factorial_ablation/common.py (V_MEAN, V_STD, horizon bin).

모델 출력 → 최종 값 (fc_predict.Pipeline.run / assemble 과 같은 계산):
  g01   = mean_s( G01_s(num) )            * V_STD + V_MEAN                       [4]   (24/48/72/96 h)
  p2o   = mean_s( P2o_s(num) )            * V_STD + V_MEAN                       [4]   (출력만)
  p_occ = mean_s( sigmoid(O_s(num).occ_logits) )                                 [16]
  p17   = mean_s( softmax(T2_s(num).onset_logits) )                              [17]  (16 bin + 사건 없음)
  t2_speed = mean_s( T2_s(num).speed )    * V_STD + V_MEAN
  r_s   = RegimeNetCF_s(images, 0, 0, persist, recur, ch_vec)                          (CME 자리 2 개는 0)
  router16 = mean_s( (r_s.anchor + Σ_k softmax(RouterR2_s(masked, 0, 0, persist, recur, fs))_k · r_s.dv_k) * V_STD + V_MEAN )
            (Router 와 RegimeNetCF 는 같은 seed 끼리 짝)
  v       = W_G01 · g01 + W_ROUTER · router16[HORIZON_BINS]
  v_final = v + BOXCOX_ALPHA · p_occ[HORIZON_BINS] · (BoxCox_C(v) − v)
  fallback (영상·PFSS 입력 하나라도 없음): v = g01, BoxCox_A 사용
  onset_bin = argmax(p17[:16]) if 1 − p17[16] ≥ T2_TAU else 16 (없음); 창 = [t0 + 6·bin h, t0 + 6·bin + 6 h)

BoxCox: cr_boxcox.BoxCoxDistributionMap.from_params(p).transform(v) — horizon 마다 BOXCOX_C[j] 하나 (j = 0..3).
f_s:    z = (clip(log10(max(fsq, FS_FLOOR) + FS_EPS), FS_CLIP_LO, FS_CLIP_HI) − FS_MEAN) / FS_STD;
        추적 실패·열린 선 없음·NaN 이면 z = 0 이고 flag = 1  → Router 입력 [z0..z4, flag]
"""

# ── 공통 ──
FINAL_MODEL = "C+HBC0.25"
SEEDS = (42, 43, 44, 45, 46)
HORIZONS_H = (24, 48, 72, 96)
HORIZON_BINS = (3, 7, 11, 15)          # 16 bin 출력에서 horizon 위치 (bin k = [t0 + 6k, t0 + 6k + 6) h 평균)
N_BINS = 16
V_MEAN, V_STD = 430.0, 100.0          # 속도 정규화 (V − V_MEAN) / V_STD

# ── 앙상블·보정 (final_selection.json) ──
W_G01, W_ROUTER = 0.5, 0.5            # base C
ROUTER_VARIANT = "R2"                # PFSS f_s 입력 Router
BOXCOX_ALPHA = 0.25                  # HSS_BoxCox_a0.25

# ── Router 입력 f_s 표준화 (ch_pfss.h5 feat_fs 속성, train 으로 정한 값) ──
FS_QUANTILES = (10, 25, 50, 75, 90)
FS_LOG10 = True
FS_EPS = 0.0
FS_FLOOR = 1e-12
FS_CLIP_LO = (0.09822601013645359, 0.27793100184761854, 0.4472282774231815, 0.5763699185604046, 0.6577859083841406)
FS_CLIP_HI = (2.077801046916467, 2.087483665225207, 2.1294679130033574, 2.2146807115941103, 2.3300780090127846)
FS_MEAN = (0.7688122922868508, 0.8761323241534791, 1.000932466998531, 1.1420157909221804, 1.2942304694191256)
FS_STD = (0.3701738196167011, 0.33906620429613654, 0.31905199372749415, 0.3137437424658866, 0.31258279998434185)

# ── HSS Box–Cox 사상 (horizon 별, BoxCoxDistributionMap.params()) ──
BOXCOX_C = [
    {"lambda_x": -1.7563264120570194, "lambda_y": -1.1967494953380247, "mu_x": 0.5693546057243677, "mu_y": 0.8349333804333353, "sigma_x": 4.217608362352808e-06, "sigma_y": 0.00015293258305136757, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7780, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +24 h
    {"lambda_x": -2.045547781138659, "lambda_y": -1.2026203528362942, "mu_x": 0.48886428286537525, "mu_y": 0.8308805764136774, "sigma_x": 6.279490954878452e-07, "sigma_y": 0.00014759266593345797, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7768, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +48 h
    {"lambda_x": -2.1677639137619487, "lambda_y": -1.1907729416471318, "mu_x": 0.46130379489422674, "mu_y": 0.8391005985796613, "sigma_x": 2.8408287446486823e-07, "sigma_y": 0.0001584663954920778, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7750, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +72 h
    {"lambda_x": -2.381539195286952, "lambda_y": -1.1772181432126385, "mu_x": 0.4198962452501872, "mu_y": 0.848703899361494, "sigma_x": 7.565470625902444e-08, "sigma_y": 0.00017156646993615805, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7743, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +96 h
]
# G01 fallback 용
BOXCOX_A = [
    {"lambda_x": -1.6764470618955905, "lambda_y": -1.1967494953380247, "mu_x": 0.5964732208173902, "mu_y": 0.8349333804333353, "sigma_x": 6.651205556134435e-06, "sigma_y": 0.00015293258305136757, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7780, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +24 h
    {"lambda_x": -2.0105851231144416, "lambda_y": -1.2026203528362942, "mu_x": 0.4973647195951758, "mu_y": 0.8308805764136774, "sigma_x": 7.645104838843978e-07, "sigma_y": 0.00014759266593345797, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7768, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +48 h
    {"lambda_x": -2.0947772628811854, "lambda_y": -1.1907729416471318, "mu_x": 0.4773760315567436, "mu_y": 0.8391005985796613, "sigma_x": 4.3467417445452186e-07, "sigma_y": 0.0001584663954920778, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7750, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +72 h
    {"lambda_x": -2.1803482123406814, "lambda_y": -1.1772181432126385, "mu_x": 0.4586413677003365, "mu_y": 0.848703899361494, "sigma_x": 2.505727163210467e-07, "sigma_y": 0.00017156646993615805, "min_train": 242.5683135986328, "max_train_obs": 777.67138671875, "n_fit": 7743, "n_nonpositive_fit": 0, "out_of_domain_policy": 'clip'},   # +96 h
]

# ── T2 onset (model_preds_info.json) ──
T2_TAU_PER_SEED = (0.2676457464694977, 0.3096209168434143, 0.27058446407318115, 0.29837584495544434, 0.3237229287624359)
T2_TAU = 0.2939899802207947            # 5 seed 평균 (fc_predict 와 같음)

# ── 가중치 (final_bundle 기준 상대 경로) ──
WEIGHTS = {
    "G01": (
        "weights/G01/G01_s42.pt",
        "weights/G01/G01_s43.pt",
        "weights/G01/G01_s44.pt",
        "weights/G01/G01_s45.pt",
        "weights/G01/G01_s46.pt",
    ),
    "RegimeNetCF": (
        "weights/RegimeNetCF/RegimeNetCF_s42.pt",
        "weights/RegimeNetCF/RegimeNetCF_s43.pt",
        "weights/RegimeNetCF/RegimeNetCF_s44.pt",
        "weights/RegimeNetCF/RegimeNetCF_s45.pt",
        "weights/RegimeNetCF/RegimeNetCF_s46.pt",
    ),
    "RouterR2": (
        "weights/RouterR2/RouterR2_s42.pt",
        "weights/RouterR2/RouterR2_s43.pt",
        "weights/RouterR2/RouterR2_s44.pt",
        "weights/RouterR2/RouterR2_s45.pt",
        "weights/RouterR2/RouterR2_s46.pt",
    ),
    "O": (
        "weights/O_occupancy/O_s42.pt",
        "weights/O_occupancy/O_s43.pt",
        "weights/O_occupancy/O_s44.pt",
        "weights/O_occupancy/O_s45.pt",
        "weights/O_occupancy/O_s46.pt",
    ),
    "P2o": (
        "weights/loaded_not_in_final/P2o/P2o_s42.pt",
        "weights/loaded_not_in_final/P2o/P2o_s43.pt",
        "weights/loaded_not_in_final/P2o/P2o_s44.pt",
        "weights/loaded_not_in_final/P2o/P2o_s45.pt",
        "weights/loaded_not_in_final/P2o/P2o_s46.pt",
    ),
    "T2": (
        "weights/loaded_not_in_final/T2/T2_s42.pt",
        "weights/loaded_not_in_final/T2/T2_s43.pt",
        "weights/loaded_not_in_final/T2/T2_s44.pt",
        "weights/loaded_not_in_final/T2/T2_s45.pt",
        "weights/loaded_not_in_final/T2/T2_s46.pt",
    ),
}
# 로더: G01·P2o = AblationNet(**ck['kwargs']) + ck['model'];  O·T2 = MultiHeadNet(**ck['kwargs']) + ck['model']
#       RegimeNetCF = RegimeNetCF(**ck['model_kwargs']) + ck['state_dict'];  RouterR2 = model.MoENet(**ck['model_kwargs']) + ck['state_dict']
WEIGHTS_SHA256 = {
    "weights/G01/G01_s42.pt": "1709d8c8cd2a57f1c258546d61f28bc85c8b902234668faff3b095a1c3c57665",
    "weights/G01/G01_s43.pt": "26e51db2b133fabda83c2171f1108cc78b33912d21c7e16476e94d34adb4ac80",
    "weights/G01/G01_s44.pt": "a16902c91b365146761841b6543531941a717149e94c2c64dea1a4e9b2dd838b",
    "weights/G01/G01_s45.pt": "bbd3e4fa32cb995aef0d539c51cce8121bbecfd673bbab206b0bb261e1ac5d3b",
    "weights/G01/G01_s46.pt": "e244d49fdba1e2a7b916cb80bf2e51cbea7611e86080cd98f7cefd3ffbbe235a",
    "weights/O_occupancy/O_s42.pt": "859e00d2ddc49ff936957a71baa09527ceb71740e73f824f756f6ba115be6ba4",
    "weights/O_occupancy/O_s43.pt": "a12ec0ee2125064df38b82cd620add2168f7c0b4217a02576db6513829579804",
    "weights/O_occupancy/O_s44.pt": "4a3c73a458498b61663257bd71eeb39435fdc3a24fcb099e4853ed25fbd4d9af",
    "weights/O_occupancy/O_s45.pt": "cc8a6a1d73ab188884aa43e7d2d5e0c6dbc541554f6b56ad3dfd562cde8c86f8",
    "weights/O_occupancy/O_s46.pt": "49b29cd8c3c4307901770cfb64aa28d580305df7cc5576898863ebd4447cafbb",
    "weights/RegimeNetCF/RegimeNetCF_s42.pt": "0dd514d368c8a953b3b4bb613b2b0e618455f31c9c99361d02a8de3a7e0b8d24",
    "weights/RegimeNetCF/RegimeNetCF_s43.pt": "580b81d3be279c2e4a3e8bb56af6e60615839ac5955b901293f156eefc41aa8f",
    "weights/RegimeNetCF/RegimeNetCF_s44.pt": "f0a46a8158b6f3e5e6c8e3ce9ba9dcec8a19d537690705f9bd01f2e2549ae65e",
    "weights/RegimeNetCF/RegimeNetCF_s45.pt": "60d260b3f8c35a40eb0b61a91d09f11eb0e0fc007d758d556aea4cbbc849ef4d",
    "weights/RegimeNetCF/RegimeNetCF_s46.pt": "2f01ed3f1a82f108b810acf293d2151b31b0c3a9bb44dbbafb2a8639462e9d86",
    "weights/RouterR2/RouterR2_s42.pt": "f5b112487a4fa77b53736fc4136c62205f0b069c40854f9b8ce7621f65bf2131",
    "weights/RouterR2/RouterR2_s43.pt": "45aaa098a2d3559fe2f263f767468c1d25b31d3808c029f88e7f7e13a9e9497e",
    "weights/RouterR2/RouterR2_s44.pt": "7811d6af587d63291e1a943f91be22ad9b99d4fe1ac92bb991e863d839c61c0a",
    "weights/RouterR2/RouterR2_s45.pt": "4d079529dbbb171853d485001e3ac9b955c98213892e8910c7f4792d86ff91a5",
    "weights/RouterR2/RouterR2_s46.pt": "25c3612bc43cb4f28864227553307458795116d97352cc868846257916472cfa",
    "weights/loaded_not_in_final/P2o/P2o_s42.pt": "dff37cede2032b6d5335f4853c0ffb8a1d383142e4b18a58eeee5c2cde0c4942",
    "weights/loaded_not_in_final/P2o/P2o_s43.pt": "038ea5ef014a1966daeaa5db8cc64a4e4437b3469983f4ce9743329f62f253bf",
    "weights/loaded_not_in_final/P2o/P2o_s44.pt": "fac68a12d6323c6272f531d027f432017eec6bfd191f2ddc12bcc6e7d7b145d8",
    "weights/loaded_not_in_final/P2o/P2o_s45.pt": "1245893a16a14b86079bf62ffaec161b71b58eb7b24700a5243c8245f7898ed9",
    "weights/loaded_not_in_final/P2o/P2o_s46.pt": "1cdccb46df0adb42728368d790a6d805802ae0e70614132005cdaa9047e46ad8",
    "weights/loaded_not_in_final/T2/T2_s42.pt": "2b5646c26bdf0ea98fa4d13e2051a3c83f6d382ab16b564ac56275c8ab55a9d2",
    "weights/loaded_not_in_final/T2/T2_s43.pt": "e3f1a995672b20ed94e0d9ab65c2da07fba64d64ad3622faae4f332cd61f0e95",
    "weights/loaded_not_in_final/T2/T2_s44.pt": "17f80691f318083e8d453e3e1d6d389c1ea33906d9cc9e71bc614a5e9317a41d",
    "weights/loaded_not_in_final/T2/T2_s45.pt": "28564af005477d562595d257e4c28d2a60775a6be6b950be4f88e97f20d29ea7",
    "weights/loaded_not_in_final/T2/T2_s46.pt": "44dbe84d32bd43b16bf5a2e1a31cabf19c22990c102184c49035509123463895",
}
