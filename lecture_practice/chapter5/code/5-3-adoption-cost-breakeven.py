"""
5-3. 도입 판정의 손익분기 — 이 업무에 CNN을 넣을 것인가
=======================================================
분석 1(`5-2`)이 낸 것은 성능 격차다. 그런데 "CNN이 Macro-F1을 0.03 더 낸다"는
문장만으로는 아무 결정도 내릴 수 없다. 결정에 필요한 것은 컷라인이다 —
**판독 면적이 얼마를 넘으면 CNN이 값을 하는가.**

이 스크립트는 정확도를 돈으로 바꾼다. 세 경로를 나란히 놓는다.

  경로 A 사람 판독      : 인건비가 매 주기 그대로 든다. 초기 비용이 없다
  경로 B 지수 규칙 자동화 : 개발이 싸다. 대신 오분류가 많아 확인 비용이 든다
  경로 C CNN 자동화     : 개발·라벨이 비싸다. 대신 오분류가 가장 적다

성능 수치는 **분석 1의 실측을 그대로 읽어 온다**(`results/ch5_perf_handoff.csv`).
비용을 맞추려고 성능을 새로 지어내지 않는다. 두 분석이 한 파이프라인이다.

가정의 등급
----------
  [검증값] 출처가 있는 공개 수치. 출처를 나란히 적는다
  [산출값] 검증값에서 계산해 얻은 값
  [가정]   근거를 찾지 못해 정한 값. 반드시 민감도로 방어한다

임계·단가를 하나 정해 놓고 결론을 내리지 않는다. 판정이 뒤집히는 지점을
파라미터마다 찾아 함께 보고한다.

실행 (프로젝트 루트, 통합 .venv):
    python lecture_practice/chapter5/code/5-3-adoption-cost-breakeven.py
    # 먼저: python lecture_practice/chapter5/code/5-2-baseline-vs-cnn.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULT_DIR = Path(__file__).resolve().parent.parent / "results"
HANDOFF = RESULT_DIR / "ch5_perf_handoff.csv"

print("=" * 66)
print("도입 판정의 손익분기: 이 업무에 CNN을 넣을 것인가")
print("=" * 66)

if not HANDOFF.exists():
    raise SystemExit(
        f"[중단] {HANDOFF.name}이 없다. 먼저 5-2-baseline-vs-cnn.py를 실행해\n"
        "        분석 1의 실측 성능을 만들어야 한다. 성능을 가정으로 채우지 않는다."
    )

# ============================================================
# 1. 분석 1에서 넘어온 실측 성능
# ============================================================
perf = pd.read_csv(HANDOFF).iloc[0]
ACC_RULE_FIXED = float(perf["규칙_정확도"])     # R1 고정 임계 규칙
ACC_BASE = float(perf["기준선_정확도"])          # 최고 저비용 기준선
ACC_CNN = float(perf["CNN_정확도"])
BASE_NAME = str(perf["최고_기준선"]).strip()
N_LABEL = int(perf["패치_수"])                   # 학습에 쓴 라벨 패치 수

print("\n--- 1. 분석 1에서 넘어온 실측 성능 (지어낸 값 없음) ---")
print(f"  고정 임계 규칙 정확도 : {ACC_RULE_FIXED:.3f}")
print(f"  최고 저비용 기준선     : {ACC_BASE:.3f}  ({BASE_NAME})")
print(f"  CNN 정확도            : {ACC_CNN:.3f}")
print(f"  CNN − 기준선 격차      : {ACC_CNN - ACC_BASE:+.3f}")
print(f"  학습에 쓴 라벨 패치     : {N_LABEL}장")

# ============================================================
# 2. 파라미터 — 등급을 붙여 명시한다
# ============================================================
# [산출값] 판독 단위. 32픽셀 × 10m = 320m 한 변 → 0.1024 km²
PATCH_M = 32 * 10
PATCH_KM2 = (PATCH_M / 1000) ** 2

# [검증값] 2025년 적용 SW기술자 평균임금 (한국소프트웨어산업협회, 2024-12-03 공표)
#   IT지원기술자 월평균 5,058,021원. 일평균 = 월평균 ÷ 20.6일(협회 산식)
#   https://www.sw.or.kr/site/sw/ex/board/View.do?cbIdx=304&bcIdx=61152
#   주의: 이것은 IT 인력 단가이지 위성영상 판독원 직군의 단가가 아니다.
#   판독 인건비의 대리값으로 쓰는 것은 [가정]이며 민감도로 방어한다.
WAGE_MONTH = 5_058_021
WORK_DAYS = 20.6
WAGE_DAY = WAGE_MONTH / WORK_DAYS          # [산출값] 245,535원/일
WAGE_HOUR = WAGE_DAY / 8                   # [산출값] 30,692원/시간

# [검증값] Sentinel-2는 Copernicus 개방 정책에 따라 무료다
IMG_FREE = 0.0
# [검증값(범위)] 상업 VHR(<1m) 아카이브 $15~30/km², 신규 촬영 $40~60/km²
#   (최소 주문 각각 25km², 100km²) — 2025 위성영상 가격 안내
#   https://ongeo-intelligence.com/blog/satellite-imagery-pricing-guide
VHR_USD_LO, VHR_USD_HI = 15.0, 30.0
FX = 1400.0                                # [가정] 환율 1,400원/USD
VHR_KRW = (VHR_USD_LO + VHR_USD_HI) / 2 * FX   # [산출값] 원/km²

# [가정] 아래 넷은 공개 단가를 찾지 못했다. 전부 민감도로 방어한다.
MIN_PER_PATCH = 2.0            # 판독원이 패치 1건을 판정하는 데 걸리는 시간(분)
ACC_HUMAN = 0.98               # 사람 판독 정확도 (사람도 경계·혼합에서 틀린다)
COST_ERR = 50_000              # 오분류 1건이 유발하는 후속 처리비(현장 확인 등)
COST_LABEL = 1_500             # 라벨 1장 단가
DEV_DAYS_RULE = 3              # 지수 규칙 구축 인일
DEV_DAYS_CNN = 15              # CNN 학습 파이프라인 구축 인일

print("\n--- 2. 파라미터와 등급 ---")
rows = [
    ("판독 단위 면적", f"{PATCH_KM2:.4f} km² (32px × 10m)", "[산출값]"),
    ("IT 인력 일평균임금", f"{WAGE_DAY:,.0f}원", "[검증값] KOSA 2025 적용 공표"),
    ("시간당 인건비", f"{WAGE_HOUR:,.0f}원", "[산출값] 일평균÷8"),
    ("Sentinel-2 영상비", "0원 (Copernicus 개방)", "[검증값]"),
    ("상업 VHR 영상비", f"{VHR_KRW:,.0f}원/km² ($15~30)", "[검증값(범위)]+[가정] 환율"),
    ("판독 1건 소요", f"{MIN_PER_PATCH:.1f}분", "[가정]"),
    ("사람 판독 정확도", f"{ACC_HUMAN:.3f}", "[가정]"),
    ("오분류 1건 처리비", f"{COST_ERR:,}원", "[가정]"),
    ("라벨 1장 단가", f"{COST_LABEL:,}원", "[가정]"),
    ("구축 인일 (규칙/CNN)", f"{DEV_DAYS_RULE}일 / {DEV_DAYS_CNN}일", "[가정]"),
]
for name, val, grade in rows:
    print(f"  {name:20s} {val:>28s}   {grade}")


# ============================================================
# 3. 비용 모형
# ============================================================
def setup_cost(dev_days, n_label, cost_label=None, wage_day=None):
    """일회성 비용 = 구축 인건비 + 라벨링 비용"""
    cost_label = COST_LABEL if cost_label is None else cost_label
    wage_day = WAGE_DAY if wage_day is None else wage_day
    return dev_days * wage_day + n_label * cost_label


def cycle_cost(area_km2, acc, per_patch_labor, img_krw_per_km2, cost_err):
    """주기 1회 비용 = 인건비 + 영상비 + 오분류 처리비"""
    n = area_km2 / PATCH_KM2
    return n * per_patch_labor + area_km2 * img_krw_per_km2 + n * (1 - acc) * cost_err


LABOR_HUMAN = MIN_PER_PATCH / 60 * WAGE_HOUR    # [산출값] 판독 1건 인건비
LABOR_AUTO = 0.0                                 # 추론 인건비는 무시할 수준
print(f"\n  [산출값] 판독 1건 인건비 = {LABOR_HUMAN:,.0f}원 "
      f"(= {MIN_PER_PATCH}분 × {WAGE_HOUR:,.0f}원/시간)")


def total_cost(path, area_km2, cycles, **ov):
    """경로별 총비용(일회성 + 주기 × 주기비용)"""
    ce = ov.get("cost_err", COST_ERR)
    cl = ov.get("cost_label", COST_LABEL)
    lab = ov.get("labor_human", LABOR_HUMAN)
    img = ov.get("img", IMG_FREE)
    if path == "human":
        return cycles * cycle_cost(area_km2, ACC_HUMAN, lab, 0.0, ce)
    if path == "rule":
        return (setup_cost(ov.get("dev_rule", DEV_DAYS_RULE), 0, cl)
                + cycles * cycle_cost(area_km2, ACC_BASE, LABOR_AUTO, img, ce))
    if path == "cnn":
        return (setup_cost(ov.get("dev_cnn", DEV_DAYS_CNN), N_LABEL, cl)
                + cycles * cycle_cost(area_km2, ACC_CNN, LABOR_AUTO, img, ce))
    raise ValueError(path)


def breakeven_area(path_a, path_b, cycles, **ov):
    """path_b가 path_a보다 싸지는 최소 면적(km²). 불가능하면 None."""
    ce = ov.get("cost_err", COST_ERR)
    cl = ov.get("cost_label", COST_LABEL)
    lab = ov.get("labor_human", LABOR_HUMAN)
    img = ov.get("img", IMG_FREE)
    spec = {
        "human": (0.0, lab, ACC_HUMAN, 0.0),
        "rule": (setup_cost(ov.get("dev_rule", DEV_DAYS_RULE), 0, cl),
                 LABOR_AUTO, ACC_BASE, img),
        "cnn": (setup_cost(ov.get("dev_cnn", DEV_DAYS_CNN), N_LABEL, cl),
                LABOR_AUTO, ACC_CNN, img),
    }
    s_a, l_a, acc_a, img_a = spec[path_a]
    s_b, l_b, acc_b, img_b = spec[path_b]
    # 면적당 주기 비용 차이 (a − b). 양수여야 넓어질수록 b가 유리해진다.
    per_km2 = ((l_a - l_b) + ce * ((1 - acc_a) - (1 - acc_b))) / PATCH_KM2 + (img_a - img_b)
    denom = cycles * per_km2
    if denom <= 0:
        return None                       # 넓혀도 역전되지 않는다
    return (s_b - s_a) / denom


# ============================================================
# 4. 손익분기 — 판독 면적이 얼마를 넘어야 값을 하는가
# ============================================================
print("\n--- 3. 손익분기 면적 (Sentinel-2 무료 영상 기준) ---")
print("  '얼마나 넓게, 몇 번 반복해야 자동화가 사람 판독보다 싸지는가'")
print(f"  {'반복 주기':>8s} | {'사람→규칙':>14s} | {'사람→CNN':>14s} | {'규칙→CNN':>14s}")
be_rows = []
for cycles in (1, 4, 12, 52):
    a1 = breakeven_area("human", "rule", cycles)
    a2 = breakeven_area("human", "cnn", cycles)
    a3 = breakeven_area("rule", "cnn", cycles)
    fmt = lambda v: "역전 없음" if v is None else f"{v:,.2f} km²"
    print(f"  {cycles:6d}회 | {fmt(a1):>14s} | {fmt(a2):>14s} | {fmt(a3):>14s}")
    be_rows.append({"반복_주기": cycles, "사람_규칙_km2": a1,
                    "사람_CNN_km2": a2, "규칙_CNN_km2": a3})

print("\n  참고: 판독 단위 1건이 0.1024 km², 시군구 하나가 대략 수백 km²,")
print("        남한 전체가 약 100,000 km² 규모다.")

print("\n--- 4. 면적별 총비용 (연 12회 반복 기준, 백만원) ---")
CYCLES = 12
print(f"  {'면적(km²)':>10s} | {'사람 판독':>10s} | {'규칙':>10s} | {'CNN':>10s} | 가장 싼 경로")
cost_rows = []
for A in (1, 10, 100, 500, 1_000, 10_000, 100_000):
    c = {p: total_cost(p, A, CYCLES) for p in ("human", "rule", "cnn")}
    best = min(c, key=c.get)
    label = {"human": "사람 판독", "rule": "규칙", "cnn": "CNN"}[best]
    print(f"  {A:10,d} | {c['human'] / 1e6:10,.1f} | {c['rule'] / 1e6:10,.1f} | "
          f"{c['cnn'] / 1e6:10,.1f} | {label}")
    cost_rows.append({"면적_km2": A, "사람_원": c["human"], "규칙_원": c["rule"],
                      "CNN_원": c["cnn"], "최저비용_경로": label})

# ============================================================
# 5. 유료 고해상도 영상은 언제 값을 하는가 (면적과 무관한 닫힌 형태)
# ============================================================
# 이득 = (패치 수) × Δ정확도 × 오분류비, 비용 = 면적 × 영상단가
# 면적이 양쪽에 같이 들어가 상쇄되므로 최소 성능 격차는 면적과 무관하다.
print("\n--- 5. 유료 고해상도 영상이 값을 하는 최소 성능 격차 ---")
print("  이득 = 패치수 × Δ정확도 × 오분류비,  비용 = 면적 × 영상단가")
print("  두 식에 면적이 같이 들어가 상쇄된다 → 최소 격차는 면적과 무관하다.")
print(f"  {'영상 단가':>22s} | {'원/km²':>12s} | {'필요 Δ정확도':>12s} | 판정")
vhr_rows = []
for usd, tag in ((15.0, "VHR 아카이브 하단"), (22.5, "VHR 아카이브 중앙"),
                 (30.0, "VHR 아카이브 상단"), (50.0, "VHR 신규 촬영")):
    krw = usd * FX
    need = krw * PATCH_KM2 / COST_ERR
    headroom = 1.0 - ACC_CNN
    verdict = "달성 불가" if need > headroom else "가능"
    print(f"  {tag:>22s} | {krw:12,.0f} | {need:11.3f} | {verdict} "
          f"(남은 여지 {headroom:.3f})")
    vhr_rows.append({"영상": tag, "USD_per_km2": usd, "원_per_km2": krw,
                     "필요_델타정확도": need, "남은_여지": headroom,
                     "판정": verdict})
print(f"  현재 CNN 정확도가 {ACC_CNN:.3f}이므로 이론상 남은 개선 여지가 "
      f"{1 - ACC_CNN:.3f}뿐이다.")
print("  주의: 유료 영상으로 정확도가 실제로 얼마나 오르는지는 재지 않았다.")
print("        이 계산은 '얼마나 올라야 본전인가'를 정할 뿐이다.")

# ============================================================
# 6. 민감도 — 판정이 뒤집히는 지점
# ============================================================
print("\n--- 6. 민감도: 어느 값에서 판정이 뒤집히는가 (연 12회 기준) ---")
print("  '사람→규칙'이 역전 없음이면 지수 규칙 자동화는 아무리 넓혀도 손해다.")
print(f"  {'파라미터':22s} | {'값':>12s} | {'사람→규칙':>14s} | {'사람→CNN':>14s} | "
      f"{'규칙→CNN':>14s}")
sens_rows = []


def sens(label, value_str, **ov):
    a1 = breakeven_area("human", "rule", CYCLES, **ov)
    a2 = breakeven_area("human", "cnn", CYCLES, **ov)
    a3 = breakeven_area("rule", "cnn", CYCLES, **ov)
    fmt = lambda v: "역전 없음" if v is None else f"{v:,.1f} km²"
    print(f"  {label:22s} | {value_str:>12s} | {fmt(a1):>14s} | {fmt(a2):>14s} | "
          f"{fmt(a3):>14s}")
    sens_rows.append({"파라미터": label, "값": value_str, "사람_규칙_km2": a1,
                      "사람_CNN_km2": a2, "규칙_CNN_km2": a3})


sens("기준(위 설정)", "—")
for v in (10_000, 50_000, 200_000, 500_000):
    sens("오분류 처리비", f"{v:,}원", cost_err=v)
for v in (0.5, 2.0, 5.0):
    sens("판독 1건 소요", f"{v}분", labor_human=v / 60 * WAGE_HOUR)
for v in (500, 1_500, 5_000):
    sens("라벨 1장 단가", f"{v:,}원", cost_label=v)
for v in (5, 15, 40):
    sens("CNN 구축 인일", f"{v}일", dev_cnn=v)

# ============================================================
# 7. 퇴화 검사 — 계산이 제정신인지 본다
# ============================================================
print("\n--- 7. 퇴화 검사 ---")
_acc_cnn_saved = ACC_CNN
ACC_CNN = ACC_BASE                      # 성능 격차를 0으로 만든다
a_deg = breakeven_area("rule", "cnn", CYCLES)
print(f"  (a) CNN 정확도를 기준선과 같게({ACC_BASE:.3f}) 두면 "
      f"규칙→CNN 손익분기: {'역전 없음' if a_deg is None else f'{a_deg:,.1f} km²'}")
print("      → 성능이 같은데 개발·라벨만 더 드는 경로이므로 '역전 없음'이 맞다.")
ACC_CNN = _acc_cnn_saved

_ce_saved = COST_ERR
a_deg2 = breakeven_area("rule", "cnn", CYCLES, cost_err=0)
print(f"  (b) 오분류 처리비를 0원으로 두면 규칙→CNN 손익분기: "
      f"{'역전 없음' if a_deg2 is None else f'{a_deg2:,.1f} km²'}")
print("      → 틀려도 비용이 없으면 정확도를 살 이유가 없다. '역전 없음'이 맞다.")

a_deg3 = breakeven_area("human", "rule", CYCLES, labor_human=0.0)
print(f"  (c) 사람 인건비를 0원으로 두면 사람→규칙 손익분기: "
      f"{'역전 없음' if a_deg3 is None else f'{a_deg3:,.1f} km²'}")
print("      → 공짜 인력이 더 정확하면 자동화할 이유가 없다. '역전 없음'이 맞다.")

# ============================================================
# 8. 저장과 판정 요약
# ============================================================
RESULT_DIR.mkdir(exist_ok=True)
pd.DataFrame(be_rows).to_csv(RESULT_DIR / "ch5_breakeven_area.csv",
                             index=False, encoding="utf-8-sig")
pd.DataFrame(cost_rows).to_csv(RESULT_DIR / "ch5_total_cost.csv",
                               index=False, encoding="utf-8-sig")
pd.DataFrame(sens_rows).to_csv(RESULT_DIR / "ch5_cost_sensitivity.csv",
                               index=False, encoding="utf-8-sig")
pd.DataFrame(vhr_rows).to_csv(RESULT_DIR / "ch5_imagery_threshold.csv",
                              index=False, encoding="utf-8-sig")
print("\n  저장: ch5_breakeven_area.csv, ch5_total_cost.csv, "
      "ch5_cost_sensitivity.csv, ch5_imagery_threshold.csv")

print("\n--- 8. 판정 ---")
# 왜 이런 결과가 나오는지는 패치 1건의 손익으로 보면 바로 드러난다.
saved = LABOR_HUMAN                                    # 자동화로 아끼는 인건비
add_rule = (ACC_HUMAN - ACC_BASE) * COST_ERR           # 규칙이 더 내는 오분류 비용
add_cnn = (ACC_HUMAN - ACC_CNN) * COST_ERR             # CNN이 더 내는 오분류 비용
print("  [패치 1건의 손익] 자동화는 인건비를 아끼는 대신 오분류 비용을 더 낸다.")
print(f"    아끼는 인건비           : {saved:>10,.0f}원")
print(f"    규칙이 더 내는 오분류 비용 : {add_rule:>10,.0f}원 "
      f"(정확도 {ACC_HUMAN:.3f} → {ACC_BASE:.3f})")
print(f"    CNN이 더 내는 오분류 비용 : {add_cnn:>10,.0f}원 "
      f"(정확도 {ACC_HUMAN:.3f} → {ACC_CNN:.3f})")
print(f"    → 규칙 순손익 {saved - add_rule:+,.0f}원/건, "
      f"CNN 순손익 {saved - add_cnn:+,.0f}원/건")

be12_hr = breakeven_area("human", "rule", 12)
be12_hc = breakeven_area("human", "cnn", 12)
be12_rc = breakeven_area("rule", "cnn", 12)
if be12_hr is None:
    print("\n  지수 규칙 자동화는 **아무리 넓혀도 사람 판독보다 싸지지 않는다.**")
    print("    아끼는 인건비보다 늘어나는 오분류 처리비가 크기 때문이다.")
    print(f"    반면 CNN은 연 12회 기준 {be12_hc:,.1f} km²부터 사람 판독보다 싸진다.")
    print("  이 과업에서 도입 판정을 가른 것은 '자동화냐 아니냐'가 아니라")
    print(f"  **정확도 {ACC_BASE:.3f}과 {ACC_CNN:.3f}의 차이**다. 그 "
          f"{ACC_CNN - ACC_BASE:.3f}이 자동화를 손해에서 이득으로 뒤집는다.")
    print("  분석 1의 성능 격차가 곧 도입 판정이었다는 뜻이며, 정확도 보고에서")
    print("  멈췄다면 '규칙으로도 0.947이면 충분하다'는 반대 결론을 냈을 것이다.")
else:
    print(f"\n  연 12회 기준 손익분기: 사람→규칙 {be12_hr:,.1f} km², "
          f"사람→CNN {be12_hc:,.1f} km², 규칙→CNN {be12_rc:,.1f} km²")
print(f"  유료 고해상도 영상은 정확도를 최소 {VHR_KRW * PATCH_KM2 / COST_ERR:.3f} 올려야 "
      f"본전인데, 남은 여지가 {1 - ACC_CNN:.3f}뿐이라 이 과업에서는 값을 하지 못한다.")
print("  단, 위 판정은 오분류 처리비 5만원 가정 위에 서 있다. 민감도(6)에서 보듯")
print("  처리비가 1만원이면 규칙 자동화도 이득으로 돌아선다 — 판정이 뒤집히는 값이다.")

print("\n--- 9. 이 계산이 말하는 것과 말하지 못하는 것 ---")
print("  말하는 것: 성능 격차를 비용으로 환산하는 절차, 그 절차가 만드는 컷라인,")
print("    그리고 컷라인이 어느 파라미터에 민감한지.")
print("  말하지 못하는 것: 실제 기관의 도입 비용. 인건비를 뺀 네 파라미터가")
print("    [가정]이고, 판독 인건비도 IT 인력 단가를 대리로 쓴 것이다. 이 숫자를")
print("    특정 사업의 예산 근거로 옮겨 쓰면 안 된다. 절차를 가져가고 값은 다시 재라.")

print("\n[완료] 도입 손익분기 계산을 마쳤다.")
