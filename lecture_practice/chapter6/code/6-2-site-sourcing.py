"""
6장 실습(6.7 분석 2): 세그멘테이션 기반 개발 가능 부지 탐색 — 면적 오차가 후보 목록을 흔드는 방식
==========================================================================================
질문: "영상에서 뽑은 마스크로 개발 후보지 목록을 좁힐 때, 세그멘테이션의 오차가
      그 목록을 어떻게 망치는가. 그리고 얼마나 되돌릴 수 있는가?"

부동산 개발·투자 실무의 초기 단계에는 site sourcing이 있다. 도시 안에서 개발
가능한 유휴 부지를 찾아 실사 대상을 수십 건으로 좁히는 일이다. 이 실습은
세그멘테이션 마스크에서 출발해 다음을 계산한다.

  객체화 → 형상 피처(면적·둘레·조밀도·최소폭·접도) → 3단 개발 요건 필터 →
  면적 편향 보정(현장 실측 표본) → 대조군 3종 → 임계값의 비용 곡선 →
  실사 예산 제약 하 방문 순서

재현성 원칙: 무거운 세그멘테이션 학습 대신 그 '출력(부지 마스크)'을 미리 준비한
  데이터(6-0b-site-simdata-prep.py가 생성·저장)에서 불러온다. 객체화·피처·필터·
  보정·비용 계산은 모두 실제 계산값이다.

데이터: 교육용 시뮬레이션(512×512 래스터, 픽셀 1m=1㎡). 6-0b의 seed 20260813.
실행:
    python 6-0b-site-simdata-prep.py      # 최초 1회: 데이터 준비
    python 6-2-site-sourcing.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

# 크로스 플랫폼 한글 폰트: 사용 가능한 첫 후보 사용(macOS/Windows/Linux).
# 없으면 그림 라벨의 한글이 □로 보일 뿐 결과·수치에는 영향 없음 → 경고만 억제.
for _f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BLOCK_PATH = DATA_DIR / "urban_block.npz"
if not BLOCK_PATH.exists():
    raise SystemExit(
        f"데이터가 없습니다: {BLOCK_PATH}\n"
        "먼저 실행: python 6-0b-site-simdata-prep.py"
    )

# ---- 개발 요건(예시로 설정한 값). 법정 기준이 아니라 사업 요건이다.
#      법정 최소 접도는 건축법 제44조의 2m이며, 여기 쓰는 12m는 그보다 훨씬
#      높은 사업자 자체 기준(차량 진출입·가로 노출)을 가정한 것이다.
A_MIN = 1000.0    # 최소 면적(㎡)
L_MIN = 12.0      # 최소 도로 접면 길이(m)
W_MIN = 15.0      # 최소폭(m)

FRONT_TOL = 3.0   # 접도 판정 허용오차(m). 침식으로 경계가 물러나는 만큼을 흡수한다.
MATCH_OBJ = 0.20  # 객체 면적의 이 비율 이상 겹쳐야 '어느 필지의 객체'로 본다
MATCH_PARCEL = 0.50  # 필지 면적의 이 비율 이상 덮여야 '검출됨'으로 본다

N_FIELD = 12      # 현장 실측 표본 수
VISIT_BUDGET = 6  # 실사 예산으로 방문 가능한 후보 수
FIELD_RNG = np.random.default_rng(6)  # 표본 추출 전용 난수(데이터 생성과 분리)


# ------------------------------------------------------------ 형상 피처
def _pad_slice(sl, shape, pad=2):
    """find_objects가 준 슬라이스를 pad만큼 넓힌다(둘레·거리변환 계산용)."""
    out = []
    for s, n in zip(sl, shape):
        out.append(slice(max(0, s.start - pad), min(n, s.stop + pad)))
    return tuple(out)


def crack_perimeter(mask):
    """4-이웃 기준 둘레(㎡ 격자에서 m 단위). w×h 직사각형이면 정확히 2(w+h)."""
    nb = np.zeros(mask.shape, np.int16)
    nb[1:, :] += mask[:-1, :]
    nb[:-1, :] += mask[1:, :]
    nb[:, 1:] += mask[:, :-1]
    nb[:, :-1] += mask[:, 1:]
    return int((4 - nb)[mask].sum())


def object_table(labels, dist_road, tol=FRONT_TOL, min_area=0):
    """연결요소마다 면적·둘레·조밀도·종횡비·최소폭·접도 길이를 계산한다."""
    rows = []
    slices = ndimage.find_objects(labels)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        big = _pad_slice(sl, labels.shape)
        sub = labels[big] == i
        area = float(sub.sum())
        if area < min_area:
            continue
        perim = crack_perimeter(sub)
        rr, cc = np.nonzero(sub)
        h = rr.max() - rr.min() + 1
        w = cc.max() - cc.min() + 1
        edt = ndimage.distance_transform_edt(sub)
        boundary = sub & ~ndimage.binary_erosion(sub)
        rows.append({
            "obj_id": i,
            "area_m2": area,
            "perim_m": float(perim),
            # 조밀도 4πA/P² : 원=1, 정사각형≈0.785, 길쭉할수록 0에 가깝다
            "compactness": float(4 * np.pi * area / perim ** 2) if perim else np.nan,
            "aspect": float(max(h, w) / min(h, w)),
            # 최소폭 대리값: 내접원 지름(거리변환 최댓값×2). 격자 이산화로 약 1m 과대.
            "min_width_m": float(2 * edt.max()),
            "frontage_m": float((dist_road[big][boundary] <= tol).sum()),
            # 허용오차 없이 '도로에 직접 맞붙은' 경계만 세면 얼마가 되는가
            "frontage_strict_m": float((dist_road[big][boundary] <= 1.0).sum()),
        })
    return pd.DataFrame(rows)


def qualifies(df, a_min=A_MIN, l_min=L_MIN, w_min=W_MIN):
    return ((df["area_m2"] >= a_min) & (df["frontage_m"] >= l_min)
            & (df["min_width_m"] >= w_min))


# ------------------------------------------------------------- 대응 관계
def overlap_counts(pred_lab, true_lab):
    """예측 객체 × 정답 필지 겹침 픽셀 수 행렬."""
    n_p, n_t = int(pred_lab.max()), int(true_lab.max())
    both = (pred_lab > 0) & (true_lab > 0)
    idx = (pred_lab[both].astype(np.int64) - 1) * n_t + (true_lab[both] - 1)
    return np.bincount(idx, minlength=n_p * n_t).reshape(n_p, n_t)


def link(pred_lab, true_lab, obj_area, parcel_area):
    """객체 → 지배 필지, 필지 → 최대 객체 대응을 만든다."""
    ov = overlap_counts(pred_lab, true_lab)
    dom_parcel = np.zeros(ov.shape[0], np.int32)          # 객체별 지배 필지(0=오탐)
    for o in range(ov.shape[0]):
        t = int(ov[o].argmax())
        if ov[o, t] >= MATCH_OBJ * obj_area[o]:
            dom_parcel[o] = t + 1
    best_obj = np.zeros(ov.shape[1], np.int32)            # 필지별 최대 겹침 객체
    detected = np.zeros(ov.shape[1], bool)
    for t in range(ov.shape[1]):
        o = int(ov[:, t].argmax())
        best_obj[t] = o + 1
        detected[t] = ov[o, t] >= MATCH_PARCEL * parcel_area[t]
    return ov, dom_parcel, best_obj, detected


# ------------------------------------------------------------- 파이프라인
def build_objects(pred, dist_road, parcel_lab, parcel_area, speck_cut=50):
    """예측 마스크 → 객체 표(+정답 필지와의 대응)."""
    lab, _ = ndimage.label(pred)
    # 산발 얼룩은 개수가 많고 면적이 수 ㎡라 후보가 될 수 없다. 표에서 미리 뺀다.
    tab = object_table(lab, dist_road, min_area=speck_cut)
    n_speckle = int(lab.max()) - len(tab)
    areas = np.zeros(int(lab.max()))
    for _, r in tab.iterrows():
        areas[int(r["obj_id"]) - 1] = r["area_m2"]
    _, dom, best_obj, detected = link(lab, parcel_lab, areas, parcel_area)
    tab = tab.copy()
    tab["dom_parcel"] = [int(dom[int(i) - 1]) for i in tab["obj_id"]]
    return lab, tab, dom, best_obj, detected, n_speckle


def funnel(tab, a_col="area_m2"):
    """면적 → 접도 → 최소폭 순서로 후보가 줄어드는 깔때기."""
    n0 = len(tab)
    s1 = tab[tab[a_col] >= A_MIN]
    s2 = s1[s1["frontage_m"] >= L_MIN]
    s3 = s2[s2["min_width_m"] >= W_MIN]
    return [("객체(얼룩 제외)", n0), ("면적 ≥ 1,000㎡", len(s1)),
            ("접도 ≥ 12m", len(s2)), ("최소폭 ≥ 15m", len(s3))], s3


def miss_report(cand, good_parcels):
    """우량 부지 중 후보 목록에 오르지 못한 필지."""
    covered = set(int(p) for p in cand["dom_parcel"] if p > 0)
    return sorted(set(good_parcels) - covered)


def main():
    print("=" * 72)
    print("개발 가능 부지 탐색: 세그멘테이션 오차가 후보 목록을 흔드는 방식 (6-2, 6.7 분석 2)")
    print("=" * 72)

    d = np.load(BLOCK_PATH)
    road, building = d["road"], d["building"]
    parcel_lab = d["parcel_id"].astype(np.int32)
    pred = d["pred"]
    n_parcel = int(parcel_lab.max())
    dist_road = ndimage.distance_transform_edt(~road)

    print(f"\n무대: 512×512 래스터(픽셀 1m=1㎡, {512 * 512 / 10000:.1f}ha) | "
          f"도로 {int(road.sum()):,}㎡ | 건물 {int(building.sum()):,}㎡")
    print(f"개발 요건(예시 설정): 면적 ≥ {A_MIN:,.0f}㎡, 접도 ≥ {L_MIN:.0f}m, "
          f"최소폭 ≥ {W_MIN:.0f}m")

    # -------- [1] 정답 필지의 형상 피처와 '우량 부지' 판정 --------
    truth = object_table(parcel_lab, dist_road)
    truth["is_good"] = qualifies(truth)
    parcel_area = truth.sort_values("obj_id")["area_m2"].to_numpy()
    good_parcels = truth.loc[truth["is_good"], "obj_id"].astype(int).tolist()
    print(f"\n[1] 정답: 유휴 부지 {n_parcel}필지 중 요건을 모두 채우는 "
          f"우량 부지 {len(good_parcels)}필지")
    print(f"    탈락 사유 — 면적 미달 {int((truth['area_m2'] < A_MIN).sum())} | "
          f"맹지·접도 부족 {int((truth['frontage_m'] < L_MIN).sum())} | "
          f"최소폭 미달 {int((truth['min_width_m'] < W_MIN).sum())} (중복 포함)")

    # -------- [2] 예측 마스크의 객체화와 검출 품질 --------
    lab, tab, dom, best_obj, detected, n_speckle = build_objects(
        pred, dist_road, parcel_lab, parcel_area)
    n_phantom = int((tab["dom_parcel"] == 0).sum())
    print(f"\n[2] 예측 마스크 객체화: 연결요소 {int(lab.max())}개 중 "
          f"{n_speckle}개는 50㎡ 미만 얼룩(제외) → 분석 대상 {len(tab)}개")
    print(f"    필지 검출률 {detected.sum()}/{n_parcel} "
          f"({detected.mean():.3f}) | 정답 필지와 무관한 오탐 객체 {n_phantom}개")

    # 과분할·병합 진단
    ov = overlap_counts(lab, parcel_lab)
    pieces = {}
    for o in tab["obj_id"]:
        p = int(dom[int(o) - 1])
        if p:
            pieces.setdefault(p, []).append(int(o))
    n_split = sum(1 for p, v in pieces.items() if len(v) > 1)
    n_merge = int(sum(1 for o in tab["obj_id"]
                      if (ov[int(o) - 1] >= 0.3 * parcel_area).sum() >= 2))
    print(f"    과분할(한 필지가 여러 객체로 쪼개짐) {n_split}필지 | "
          f"병합(한 객체가 두 필지 이상을 덮음) {n_merge}객체")

    # 면적 오차: 검출된 필지 ↔ 지배 객체
    obj_area = tab.set_index("obj_id")["area_m2"].to_dict()
    err_rows = []
    for t in range(n_parcel):
        if not detected[t]:
            continue
        o = int(best_obj[t])
        if o in obj_area:
            err_rows.append({"parcel": t + 1, "true_m2": parcel_area[t],
                             "est_m2": obj_area[o],
                             "rel_err": obj_area[o] / parcel_area[t] - 1})
    edf = pd.DataFrame(err_rows)
    q1, med, q3 = edf["rel_err"].quantile([0.25, 0.5, 0.75])
    print(f"    면적 상대오차(검출 {len(edf)}필지): 중앙값 {med:+.3f} "
          f"(1사분위 {q1:+.3f} / 3사분위 {q3:+.3f}) — 침식이 면적을 체계적으로 깎는다")
    half = edf["true_m2"].median()
    big_med = edf.loc[edf["true_m2"] >= half, "rel_err"].median()
    small_med = edf.loc[edf["true_m2"] < half, "rel_err"].median()
    print(f"    오차는 크기에 따라 다르다 — 큰 절반 {big_med:+.3f} / "
          f"작은 절반 {small_med:+.3f} (둘레÷면적 비가 작은 필지에서 손실이 작다)")

    # 과분할된 필지는 면적이 조각난 채로 보고된다
    split_ids = [int(s) for s in d["split_ids"]]
    for p in split_ids:
        frag = [obj_area[o] for o in pieces.get(p, []) if o in obj_area]
        if frag:
            print(f"    과분할 필지 {p}: 참 {parcel_area[p - 1]:,.0f}㎡ → 조각 "
                  f"{len(frag)}개(최대 {max(frag):,.0f}㎡, 합 {sum(frag):,.0f}㎡)")

    # -------- [3] 접도 판정의 허용오차 --------
    strict_ok = tab[tab["frontage_strict_m"] >= L_MIN]
    strict_real = int((strict_ok["dom_parcel"] > 0).sum())
    print(f"\n[3] 접도 판정 허용오차: 도로에 직접 맞붙은 경계만 세면 분석 대상 "
          f"{len(tab)}개 중 {len(tab) - len(strict_ok)}개가 접면 부족으로 탈락한다")
    print(f"    남는 {len(strict_ok)}개 중 진짜 필지는 {strict_real}개 — "
          f"침식이 진짜 부지를 도로에서 한 겹 떼어 놓는 사이, 침식되지 않은 오탐만 "
          f"도로에 붙어 살아남는다.")
    print(f"    → 허용오차는 침식 깊이보다 커야 한다. 여기서는 {FRONT_TOL:.0f}m를 쓴다.")

    # -------- [4] 필터 깔때기 (보정 전) --------
    steps, cand_raw = funnel(tab)
    print("\n[4] 개발 요건 필터 깔때기 (면적 보정 전)")
    for name, n in steps:
        print(f"    {name:<18} {n:>4}건")
    missed_raw = miss_report(cand_raw, good_parcels)
    print(f"    후보 {len(cand_raw)}건 | 우량 부지 오탈락 {len(missed_raw)}건 "
          f"(필지 {missed_raw})")

    # -------- [5] 면적 편향 보정 --------
    # 실사 검토 풀: 접도·최소폭을 통과하고 추정 면적이 임계의 70% 이상인 객체.
    # 임계에 아슬아슬한 객체까지 넣어야 편향을 잴 표본이 한쪽으로 쏠리지 않는다.
    pool_df = tab[(tab["frontage_m"] >= L_MIN) & (tab["min_width_m"] >= W_MIN)
                  & (tab["area_m2"] >= 0.7 * A_MIN)]
    pool = pool_df["obj_id"].to_numpy()
    print(f"\n[5] 면적 편향 보정: 실사 검토 풀 {len(pool)}건 중 무작위 "
          f"{min(N_FIELD, len(pool))}건을 현장 실측한다")
    take = FIELD_RNG.choice(pool, size=min(N_FIELD, len(pool)), replace=False)
    samp = []
    n_not_site = 0
    for o in take:
        p = int(dom[int(o) - 1])
        if p == 0:                       # 현장에 가 보니 부지가 아니었던 경우
            n_not_site += 1
            continue
        samp.append({"obj": int(o), "est": obj_area[int(o)],
                     "true": parcel_area[p - 1],
                     "perim": float(tab.loc[tab["obj_id"] == o, "perim_m"].iloc[0])})
    sdf = pd.DataFrame(samp)
    k_cand = float((sdf["true"] / sdf["est"]).mean())
    c_hat = float((sdf["perim"] * (sdf["true"] - sdf["est"])).sum()
                  / (sdf["perim"] ** 2).sum())
    print(f"    실측 {len(take)}건 중 {n_not_site}건은 부지가 아님(오탐)으로 판명, "
          f"{len(sdf)}건으로 보정 계수를 추정")
    print(f"    ① 스칼라 보정  k = 평균(참/추정) = {k_cand:.3f}")
    print(f"    ② 둘레 보정   Â = A + ĉ·P,  ĉ = {c_hat:.3f} "
          f"(침식 한 겹이면 ĉ≈1이 이론값)")

    # 표본을 어디서 뽑느냐에 따라 k가 달라진다
    k_oracle = float((edf["true_m2"] / edf["est_m2"]).mean())
    real_objs = [int(o) for o in tab["obj_id"] if dom[int(o) - 1] > 0]
    take_all = FIELD_RNG.choice(real_objs, size=min(N_FIELD, len(real_objs)),
                                replace=False)
    ratios_all = [parcel_area[dom[int(o) - 1] - 1] / obj_area[int(o)]
                  for o in take_all]
    print(f"    ※ 표본을 어디서 뽑느냐로 k가 달라진다 — 실사 풀 표본 {k_cand:.3f} | "
          f"검출 필지 전수(모의 세계의 참값) {k_oracle:.3f} | "
          f"검출된 모든 객체 {np.mean(ratios_all):.3f}(중앙값 "
          f"{np.median(ratios_all):.3f})")
    print("      실사 풀 표본은 큰 필지 쪽으로 치우쳐 손실을 작게 보고, 모든 객체를 "
          "대상으로 하면 과분할 조각(참/추정≈2)이 섞여 반대로 부풀린다.")

    tab["area_scalar"] = tab["area_m2"] * k_cand
    tab["area_perim"] = tab["area_m2"] + c_hat * tab["perim_m"]

    # -------- [6] 보정 전후 오탈락 --------
    print("\n[6] 오탈락 분석 (이 예제의 결론표)")
    rows = []
    cands = {}
    for name, col in [("보정 없음", "area_m2"), ("스칼라 k 보정", "area_scalar"),
                      ("둘레 모형 보정", "area_perim")]:
        _, c = funnel(tab, a_col=col)
        cands[name] = c
        miss = miss_report(c, good_parcels)
        false_pass = int(sum(1 for p in c["dom_parcel"]
                             if int(p) == 0 or int(p) not in good_parcels))
        rows.append({"보정": name, "후보 수": len(c),
                     "우량 부지 오탈락": len(miss),
                     "우량 부지 포착": len(good_parcels) - len(miss),
                     "헛후보(비우량)": false_pass})
    mdf = pd.DataFrame(rows)
    print(mdf.to_string(index=False))
    print(f"    (전체 우량 부지 {len(good_parcels)}필지 기준)")
    deep = set(int(i) for i in d["deep_erosion_ids"])
    left = miss_report(cands["둘레 모형 보정"], good_parcels)
    print(f"    보정 뒤에도 남은 오탈락 {left} — 그중 두 겹 침식 필지 "
          f"{len([p for p in left if p in deep])}건. 전역 보정 계수 하나로는 "
          f"필지마다 다른 침식 깊이를 되돌리지 못한다.")

    # -------- [7] 대조군: 오차를 하나씩 빼 본다 --------
    print("\n[7] 대조군 — 심은 오차를 하나씩만 제거하고 같은 파이프라인을 돌린다")
    ctrl_rows = []
    for name, key in [("본 실험(오차 3종)", "pred"),
                      ("C1 침식 제거", "pred_c1_noerode"),
                      ("C2 오탐 제거", "pred_c2_nofp"),
                      ("C3 과분할 제거", "pred_c3_nosplit")]:
        lab_c, tab_c, dom_c, best_c, det_c, _ = build_objects(
            d[key], dist_road, parcel_lab, parcel_area)
        _, cand_c = funnel(tab_c)
        miss_c = miss_report(cand_c, good_parcels)
        fp_c = int(sum(1 for p in cand_c["dom_parcel"]
                       if int(p) == 0 or int(p) not in good_parcels))
        oa_c = tab_c.set_index("obj_id")["area_m2"].to_dict()
        errs = [oa_c[int(best_c[t])] / parcel_area[t] - 1
                for t in range(n_parcel)
                if det_c[t] and int(best_c[t]) in oa_c]
        ctrl_rows.append({"세계": name, "검출 필지": int(det_c.sum()),
                          "면적오차 중앙값": round(float(np.median(errs)), 3),
                          "후보 수": len(cand_c),
                          "우량 오탈락": len(miss_c), "헛후보": fp_c})
    cdf = pd.DataFrame(ctrl_rows)
    print(cdf.to_string(index=False))

    # -------- [8] 임계값의 비용 곡선 --------
    print("\n[8] 면적 임계를 어디에 둘 것인가 — 후보 수와 오탈락의 맞바꿈")
    taus = np.arange(0.30, 1.51, 0.025) * A_MIN
    cost_rows = []
    base = tab[(tab["frontage_m"] >= L_MIN) & (tab["min_width_m"] >= W_MIN)]
    for tau in taus:
        c = base[base["area_m2"] >= tau]
        miss = len(miss_report(c, good_parcels))
        cost_rows.append({"tau_m2": float(tau), "n_cand": len(c), "n_miss": miss})
    cost = pd.DataFrame(cost_rows)
    for ratio in [1, 3, 10]:
        cost[f"cost_x{ratio}"] = cost["n_cand"] + ratio * cost["n_miss"]
        best = cost.loc[cost[f"cost_x{ratio}"].idxmin()]
        print(f"    놓친 우량 부지 1건 = 실사 {ratio:>2}건 비용일 때 → "
              f"최적 임계 {best['tau_m2']:>7,.0f}㎡ "
              f"(후보 {int(best['n_cand'])}건, 오탈락 {int(best['n_miss'])}건)")
    at1000 = cost.loc[(cost["tau_m2"] - A_MIN).abs().idxmin()]
    tau_bias = A_MIN * (1 + med)
    at_bias = cost.loc[(cost["tau_m2"] - tau_bias).abs().idxmin()]
    print(f"    참고: 요건 그대로인 1,000㎡에서는 후보 {int(at1000['n_cand'])}건, "
          f"오탈락 {int(at1000['n_miss'])}건")
    print(f"    참고: 면적 편향({med:+.3f})만큼 낮춘 {at_bias['tau_m2']:,.0f}㎡에서는 "
          f"후보 {int(at_bias['n_cand'])}건, 오탈락 {int(at_bias['n_miss'])}건")
    print("    → 놓침이 실사보다 두 배만 비싸도 임계를 요건값보다 낮춰 잡는 편이 낫다. "
          "요건값을 그대로 쓰는 것은 놓침이 실사만큼 싼 세계에서만 최적이다.")

    # -------- [9] 실사 예산 제약 하 방문 순서 --------
    print(f"\n[9] 실사 예산 {VISIT_BUDGET}건 — 무엇을 먼저 볼 것인가")
    c = cands["둘레 모형 보정"].copy()
    c["slack"] = np.minimum.reduce([
        c["area_perim"] / A_MIN, c["frontage_m"] / L_MIN,
        c["min_width_m"] / W_MIN])
    c["is_good"] = [int(p) in good_parcels for p in c["dom_parcel"]]
    c["true_m2"] = [parcel_area[int(p) - 1] if int(p) > 0 else 0.0
                    for p in c["dom_parcel"]]
    print(f"    후보 {len(c)}건 중 앞의 {VISIT_BUDGET}건만 방문할 수 있다.")
    for name, key in [("면적 큰 순", "area_perim"), ("여유폭 큰 순", "slack"),
                      ("(참고) 정답을 아는 순", "is_good")]:
        top = c.sort_values(key, ascending=False).head(VISIT_BUDGET)
        hit = top[top["is_good"]]
        print(f"    {name:<16} 적중 {len(hit)}/{VISIT_BUDGET}건"
              f"(서로 다른 필지 {hit['dom_parcel'].nunique()}개) | "
              f"확보한 참 면적 "
              f"{hit.drop_duplicates('dom_parcel')['true_m2'].sum():,.0f}㎡")
    top = c.sort_values("area_perim", ascending=False).head(VISIT_BUDGET)
    waste = top[~top["is_good"]]
    n_fp = int((waste["dom_parcel"] == 0).sum())
    print(f"    헛걸음 {len(waste)}건의 정체 — 유휴 부지가 아닌 오탐 {n_fp}건, "
          f"부지이지만 요건 미달 {len(waste) - n_fp}건")
    print("    → 두 순서의 성적이 같다. 후보의 기하 피처만으로는 오탐을 가려낼 수 "
          "없기 때문이다. 순서를 바꿔 풀 문제가 아니라 분류를 고쳐야 하는 문제다.")

    # -------- [10] 산출물 저장 --------
    out = c.sort_values("slack", ascending=False).copy()
    out["visit_rank"] = range(1, len(out) + 1)
    keep = ["obj_id", "dom_parcel", "area_m2", "area_scalar", "area_perim",
            "perim_m", "compactness", "aspect", "min_width_m", "frontage_m",
            "slack", "is_good", "true_m2", "visit_rank"]
    out[keep].round(3).to_csv(RESULTS_DIR / "ch6_site_candidates.csv",
                              index=False, encoding="utf-8-sig")
    miss_final = miss_report(cands["둘레 모형 보정"], good_parcels)
    mrow = truth[truth["obj_id"].isin(miss_report(cand_raw, good_parcels))].copy()
    mrow["보정후_남은오탈락"] = mrow["obj_id"].isin(miss_final)
    mrow["두겹침식"] = mrow["obj_id"].isin(deep)
    mrow.round(3).to_csv(RESULTS_DIR / "ch6_missed_sites.csv",
                         index=False, encoding="utf-8-sig")
    cost.round(3).to_csv(RESULTS_DIR / "ch6_threshold_cost.csv",
                         index=False, encoding="utf-8-sig")
    print(f"\n[10] 저장: ch6_site_candidates.csv({len(out)}행) | "
          f"ch6_missed_sites.csv({len(mrow)}행) | ch6_threshold_cost.csv")

    # -------- [11] 지도 --------
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    base_img = np.zeros((512, 512, 3))
    base_img[...] = 0.97
    base_img[road] = [0.75, 0.75, 0.78]
    base_img[building] = [0.60, 0.62, 0.68]
    left = base_img.copy()
    good_mask = np.isin(parcel_lab, good_parcels)
    left[(parcel_lab > 0) & ~good_mask] = [0.98, 0.85, 0.55]
    left[good_mask] = [0.20, 0.55, 0.30]
    axes[0].imshow(left)
    axes[0].set_title(f"정답: 유휴 부지 {n_parcel}필지 "
                      f"(진한 색 = 요건 충족 {len(good_parcels)}필지)")

    right = base_img.copy()
    right[pred] = [0.85, 0.85, 0.90]
    cand_ids = set(int(i) for i in cands["둘레 모형 보정"]["obj_id"])
    right[np.isin(lab, list(cand_ids))] = [0.15, 0.40, 0.75]
    if miss_final:
        right[np.isin(parcel_lab, miss_final)] = [0.85, 0.15, 0.15]
    axes[1].set_title(f"예측: 최종 후보 {len(cand_ids)}건(파랑), "
                      f"놓친 우량 부지 {len(miss_final)}건(빨강)")
    axes[1].imshow(right)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig_path = RESULTS_DIR / "6-2-site-sourcing-map.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[11] 지도 저장 → {fig_path.name}")

    print("\n[완료] 세그멘테이션 오차가 후보 목록을 흔드는 방식을 계산했다.")


if __name__ == "__main__":
    main()
