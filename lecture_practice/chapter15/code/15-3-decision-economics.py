"""
15장 실습 3: 추정값을 결정으로 바꾸는 층 — 임계비·NPV·용량 제약
================================================================================
결정할 문제
--------------------------------------------------------------------------------
추정값과 예측구간은 결정이 아니다. 점추정 100, 90% 구간 [70, 130]을 받았을 때 몇 개를
발주할 것인가? 구간의 어느 값도 답이 아니다. **손실이 비대칭이면 구간의 어느 지점에서
결정이 뒤집히고, 그 지점은 돈이 정한다.**

이 스크립트는 셋을 계산한다.
  (1) 임계비와 발주량   — 같은 예측, 정반대 결정
  (2) NPV와 hurdle rate — 출점을 할 것인가, 전이율 몇 %까지 견디는가
  (3) 두 경로의 대조    — 15-2가 추정한 시장 단위 효과와 매장 단위 계산이 맞는가

(3)이 이 장의 두 축이 만나는 자리다. 15-2의 추정 편향이 여기서 **돈의 언어로** 나타난다.

가정값에 대하여
--------------------------------------------------------------------------------
비용·투자·할인율 파라미터는 전부 **가정값**이다. 실제 업종 원가를 인용할 근거가 없어
지어내지 않고 가정임을 표에 명시한다. 그래서 이 계산의 결론은 "이 값이 최적"이 아니라
**"비율이 이렇게 움직이면 결정이 여기서 뒤집힌다"**다. 절대값이 아니라 민감도가 내용이다.

3층 모델에서의 위치: 이 스크립트는 통째로 [3층]이다. 학습 모델도, 공간 연산도 없다.
앞 장들이 만든 예측·추정·구간을 **결정**으로 바꾸는 층만 다룬다.

실행:
    python 15-2-market-synthetic-control.py   # 15-2-estimate.json 생성
    python 15-3-decision-economics.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================================================
# 가정값 — 전부 가정이며, 출처가 아니라 감도 분석의 출발점이다
# ===========================================================
ASSUMPTIONS = [
    ("수요 예측 점추정", "100 단위/일", "가정 — 앞 장의 예측 모형 산출을 상정"),
    ("수요 예측 90% 구간", "[70, 130]", "가정 — conformal 예측구간을 상정"),
    ("신선식품 Cu (결품 1단위 손실)", "1,200원", "가정 — 판매 마진"),
    ("신선식품 Co (잉여 1단위 손실)", "2,000원", "가정 — 원가 + 폐기비"),
    ("내구재 Cu (결품 1단위 손실)", "15,000원", "가정 — 마진 + 기회 상실"),
    ("내구재 Co (잉여 1단위 손실)", "1,500원", "가정 — 재고 유지비(다음에 팔림)"),
    ("신규점 월 총매출", "3,000만원", "가정 — 14.6 배분 수요와 무관한 별도 가정"),
    ("기여이익률", "15%", "가정 — 임차료·인건비 차감 후"),
    ("초기 투자", "15,000만원", "가정 — 보증금·인테리어·집기"),
    ("임대차 계약 기간", "60개월", "가정"),
    ("hurdle rate", "연 12%", "가정 — 자본의 기회비용"),
    ("시장 내 동종 매장 수 N", "20개", "가정 — (3) 대조에서 결론을 좌우하는 값"),
    ("월 처리 가능 매출 상한", "3,500만원", "가정 — 좌석·주방 용량"),
]

DEMAND_MU = 100.0
DEMAND_LO, DEMAND_HI = 70.0, 130.0
COST_CASES = {
    "신선식품": {"Cu": 1200.0, "Co": 2000.0},
    "내구재": {"Cu": 15000.0, "Co": 1500.0},
}

GROSS_MONTHLY = 3000.0          # 만원
MARGIN_RATE = 0.15
CAPEX = 15000.0                 # 만원
MONTHS = 60
HURDLE_ANNUAL = 0.12
N_STORES_IN_MARKET = 20
CAPACITY_MONTHLY = 3500.0       # 만원


# ===========================================================
# 1. 임계비와 newsvendor 해
# ===========================================================
def demand_sigma(lo, hi, coverage=0.90):
    """예측구간을 정규분포의 표준편차로 되돌린다.

    구간 폭이 (hi − lo)이고 그것이 중앙 coverage를 덮는다면, 폭 = 2·z·σ다.
    정규 가정을 쓴 것이며, 분포가 비대칭이면 이 환산이 틀린다는 점을 밝혀 둔다.
    """
    z = stats.norm.ppf(0.5 + coverage / 2)
    return (hi - lo) / (2 * z)


def critical_ratio(cu, co):
    """임계비 Cu/(Cu+Co).

    한 단위를 더 시킬지 말지의 한계 판단에서 나온다. 한 단위를 더 시켰을 때 그것이
    팔릴 확률을 p라 하면, 기대 이득은 p·Cu이고 기대 손실은 (1−p)·Co다. 둘이 같아지는
    p가 Cu/(Cu+Co)이고, 그 확률을 만족시키는 재고 수준이 최적 발주량이다. 즉 임계비는
    **곧 목표 서비스 수준**이고, 최적 발주량은 수요 분포의 그 분위수다.

    분위수 해가 재고 이론에 등장한 초기 문헌은 Arrow, Harris & Marschak(1951)이다.
    다만 그 논문은 다기간 (S, s) 정책을 다루므로, 단일기간 임계비 공식 자체를 그 논문의
    결과로 소개하지 않는다. 표준 결과로 서술하고 특정 문헌에 귀속하지 않는다.
    """
    return cu / (cu + co)


def newsvendor_q(mu, sigma, cr):
    """최적 발주량 = 수요 분포의 임계비 분위수."""
    return mu + sigma * stats.norm.ppf(cr)


def expected_cost(q, mu, sigma, cu, co, n_draw=200_000, seed=42):
    """발주량 q의 기대 손실을 몬테카를로로 계산한다.

    공식이 정말 손실을 최소화하는지 수치로 확인하는 검산이다. 공식을 믿고 넘어가면
    부호 하나 틀린 것을 못 잡는다.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(mu, sigma, n_draw)
    short = np.maximum(d - q, 0.0)
    over = np.maximum(q - d, 0.0)
    return float(np.mean(cu * short + co * over))


# ===========================================================
# 2. NPV와 hurdle rate
# ===========================================================
def annuity_factor(months, monthly_rate):
    """월 1원을 months개월 받을 때의 현재가치 계수."""
    return (1 - (1 + monthly_rate) ** -months) / monthly_rate


def npv_and_payback(monthly_cf, capex, months, monthly_rate):
    """NPV와 할인 회수기간(개월). 회수 못 하면 None."""
    npv = monthly_cf * annuity_factor(months, monthly_rate) - capex
    cum, payback = 0.0, None
    for m in range(1, months + 1):
        cum += monthly_cf / (1 + monthly_rate) ** m
        if payback is None and cum >= capex:
            payback = m
    return npv, payback


def breakeven_transfer(gross, margin, capex, months, monthly_rate):
    """NPV = 0이 되는 전이율. 이 값이 곧 '견딜 수 있는 자기잠식의 상한'이다."""
    af = annuity_factor(months, monthly_rate)
    return 1.0 - capex / (gross * margin * af)


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 78)
    log("15-3 추정값을 결정으로 바꾸는 층 — 임계비·NPV·용량 제약")
    log("=" * 78)
    log()
    log("[가정값 목록] 전부 가정이며 출처가 아니다. 결론은 절대값이 아니라 민감도다.")
    log(f"{'파라미터':<28}{'값':>14}   근거")
    for name, val, src in ASSUMPTIONS:
        log(f"{name:<28}{val:>14}   {src}")
    log()

    # =======================================================
    # (1) 임계비 — 같은 예측, 정반대 결정
    # =======================================================
    log("=" * 78)
    log("(1) 비대칭 손실과 임계비 — 같은 예측에서 정반대 결정이 나온다")
    log("=" * 78)
    sigma = demand_sigma(DEMAND_LO, DEMAND_HI)
    log(f"예측: 점추정 {DEMAND_MU:.0f}, 90% 구간 [{DEMAND_LO:.0f}, {DEMAND_HI:.0f}]")
    log(f"  → 정규 가정으로 환산한 표준편차 σ = {sigma:.2f}")
    log("  (구간 폭 = 2·z·σ. 수요 분포가 비대칭이면 이 환산은 틀린다.)")
    log()
    rows = []
    for case, c in COST_CASES.items():
        cr = critical_ratio(c["Cu"], c["Co"])
        q = newsvendor_q(DEMAND_MU, sigma, cr)
        q_int = int(np.round(q))
        # 검산: 공식 해가 정말 기대손실 최소인가
        grid = np.arange(q_int - 12, q_int + 13)
        costs = np.array([expected_cost(g, DEMAND_MU, sigma, c["Cu"], c["Co"])
                          for g in grid])
        q_num = int(grid[np.argmin(costs)])
        rows.append({"구분": case, "Cu": c["Cu"], "Co": c["Co"], "임계비": cr,
                     "최적 발주량": q, "정수 발주량": q_int,
                     "수치 최소화 발주량": q_num,
                     "기대손실(원)": float(costs[np.argmin(costs)])})
        log(f"[{case}] Cu {c['Cu']:,.0f}원 / Co {c['Co']:,.0f}원")
        log(f"  임계비 = {c['Cu']:,.0f} / ({c['Cu']:,.0f} + {c['Co']:,.0f}) = {cr:.3f}")
        log(f"  최적 발주량 = 수요분포의 {cr:.1%} 분위수 = {q:.1f} → {q_int}단위")
        log(f"  검산(몬테카를로 기대손실 최소) = {q_num}단위  "
            f"{'일치' if abs(q_num - q_int) <= 1 else '**불일치 — 공식 재검토**'}")
        log()
    cr_tab = pd.DataFrame(rows)
    q_fresh = int(cr_tab.loc[cr_tab["구분"] == "신선식품", "정수 발주량"].iloc[0])
    q_dur = int(cr_tab.loc[cr_tab["구분"] == "내구재", "정수 발주량"].iloc[0])
    log(f"**같은 구간 [{DEMAND_LO:.0f}, {DEMAND_HI:.0f}]에서 한쪽은 {q_fresh}, "
        f"다른 쪽은 {q_dur}을 시킨다.** 차이 {q_dur - q_fresh}단위.")
    log("예측이 같아도 결정이 갈린다. 갈리게 만드는 것은 모형이 아니라 두 손실의 비율이다.")
    log("신선식품은 남으면 버려야 하므로(Co 큼) 보수적으로, 내구재는 못 팔면 기회를 잃고")
    log("남은 재고는 다음에 팔리므로(Cu 큼) 공격적으로 시킨다.")
    log()
    log("[임계비 훑기] 임계비가 발주량을 어떻게 옮기는가")
    sweep = []
    log(f"{'임계비':>8}{'분위수 z':>10}{'발주량':>9}")
    for cr in np.arange(0.1, 0.95, 0.1):
        q = newsvendor_q(DEMAND_MU, sigma, cr)
        sweep.append({"임계비": float(cr), "발주량": float(q)})
        log(f"{cr:>8.1f}{stats.norm.ppf(cr):>10.3f}{q:>9.1f}")
    log("임계비 0.5(두 손실이 같음)에서만 점추정이 답이다. 그 밖에서는 점추정을 쓰는 것이")
    log("곧 '두 손실이 같다'고 가정하는 셈이다 — 대개 사실이 아니다.")
    log()

    # =======================================================
    # (2) NPV와 hurdle rate
    # =======================================================
    log("=" * 78)
    log("(2) hurdle rate와 NPV — 전이율 몇 %까지 견디는가")
    log("=" * 78)
    r_m = (1 + HURDLE_ANNUAL) ** (1 / 12) - 1
    af = annuity_factor(MONTHS, r_m)
    log(f"연 {HURDLE_ANNUAL:.0%} → 월 할인율 {r_m:.5f}, {MONTHS}개월 연금계수 {af:.2f}")
    log(f"월 기여이익(전이율 0%) = {GROSS_MONTHLY:,.0f}만원 × {MARGIN_RATE:.0%} = "
        f"{GROSS_MONTHLY * MARGIN_RATE:,.0f}만원")
    log()
    npv_rows = []
    log(f"{'전이율':>8}{'월 순기여이익':>14}{'NPV(만원)':>12}{'회수기간':>10}  판정")
    for t in [0.00, 0.15, 0.25, 0.35]:
        cf = GROSS_MONTHLY * MARGIN_RATE * (1 - t)
        npv, pb = npv_and_payback(cf, CAPEX, MONTHS, r_m)
        npv_rows.append({"전이율": t, "월 순기여이익": cf, "NPV": npv,
                         "회수기간(월)": pb})
        log(f"{t:>8.0%}{cf:>14,.0f}{npv:>12,.0f}"
            f"{(str(pb) + '개월') if pb else '미회수':>10}  "
            f"{'통과' if npv > 0 else '기각'}")
    be = breakeven_transfer(GROSS_MONTHLY, MARGIN_RATE, CAPEX, MONTHS, r_m)
    log()
    log(f"**손익분기 전이율 = {be:.1%}**")
    log(f"  전이율이 {be:.1%}를 넘으면 이 출점은 자본의 기회비용을 못 넘는다.")
    log()
    log("14장이 자기잠식 상한을 30%·25% 같은 값으로 두는 관행을 소개했다. 그 숫자가")
    log("어디서 오는지가 여기서 답해진다. **상한은 관행이 아니라 손익분기 전이율에서")
    log(f"역산된 값이다.** 이 파라미터 조합에서는 {be:.1%}이고, 실무의 25%는 그보다")
    log("낮으니 안전 여유를 둔 값으로 읽힌다. 파라미터가 달라지면 상한도 달라진다 —")
    log("그러므로 '이격 거리 몇 미터', '상한 몇 %'를 보편 상수처럼 쓰면 안 된다.")
    log()
    log("[민감도] 손익분기 전이율은 어느 가정에 민감한가")
    log(f"{'변화':<26}{'손익분기 전이율':>16}")
    base = dict(gross=GROSS_MONTHLY, margin=MARGIN_RATE, capex=CAPEX,
                months=MONTHS, rate=r_m)
    variants = [
        ("기준", base),
        ("기여이익률 15% → 12%", base | {"margin": 0.12}),
        ("기여이익률 15% → 20%", base | {"margin": 0.20}),
        ("초기 투자 1.5억 → 2.0억", base | {"capex": 20000.0}),
        ("계약 60개월 → 36개월", base | {"months": 36}),
        ("hurdle 12% → 8%", base | {"rate": (1.08) ** (1 / 12) - 1}),
    ]
    sens_rows = []
    for label, p in variants:
        v = breakeven_transfer(p["gross"], p["margin"], p["capex"], p["months"], p["rate"])
        sens_rows.append({"변화": label, "손익분기 전이율": v})
        note = "  ← 음수 = 자기잠식이 0이어도 기각" if v <= 0 else ""
        log(f"{label:<26}{v:>16.1%}{note}")
    log()
    log("읽는 법 둘.")
    log("  ① 손익분기 전이율이 **음수**면 자기잠식이 전혀 없어도 이 출점은 기각이다. 계약")
    log("     기간을 60개월에서 36개월로 줄이면 그렇게 된다 — 회수 기간이 계약 기간보다")
    log("     길어지기 때문이다. 임대차 조건이 입지 판단의 일부라는 뜻이며, 자기잠식을")
    log("     계산하기 전에 먼저 확인할 값이다.")
    log("  ② 상한은 기여이익률에 극단적으로 민감하다. 15%에서 12%로 3%포인트만 내려가도")
    log("     견딜 수 있는 자기잠식이 26.9%에서 8.6%로 무너진다. 그러므로 '전이율 상한 25%'")
    log("     같은 규칙을 브랜드·업종을 가로질러 쓰면 안 된다. 원가 구조가 다르면 상한이")
    log("     다르다. 관행값을 쓰기 전에 자기 원가로 역산하는 것이 순서다.")
    log()

    # =======================================================
    # (3) 두 경로의 대조 — 15-2의 추정 편향이 돈으로 나타난다
    # =======================================================
    log("=" * 78)
    log("(3) 두 경로의 대조 — 시장 단위 추정치와 매장 단위 계산이 맞는가")
    log("=" * 78)
    est_path = RESULTS_DIR / "15-2-estimate.json"
    if not est_path.exists():
        log(f"[건너뜀] {est_path.name}이 없습니다. 먼저 15-2를 실행하세요.")
        est = None
    else:
        est = json.loads(est_path.read_text(encoding="utf-8"))
        att_hat = est["att_pct_scm_clean"] / 100.0
        att_true = float(np.expm1(est["true_effect_log"]))
        log(f"15-2의 추정: 시장 매출 {att_hat:+.2%} (placebo p={est['placebo_p']:.3f}, 유의)")
        log(f"15-2의 참값: 시장 매출 {att_true:+.2%}")
        log()
        log("두 경로가 같은 값을 가리켜야 한다.")
        log("  경로 1 (매장 단위): 순증 = 신규점 총매출 × (1 − 전이율)")
        log("  경로 2 (시장 단위): 순증 = 시장 기준 매출 × ATT = (N × 신규점 총매출) × ATT")
        log("  두 식을 같게 두면  **전이율 = 1 − N × ATT**")
        log()
        log(f"{'N (시장 내 동종 매장 수)':<24}{'참값 ATT 함의 전이율':>22}"
            f"{'추정 ATT 함의 전이율':>22}")
        recon_rows = []
        for n in [10, 15, 20, 25, 30]:
            t_true = 1 - n * att_true
            t_hat = 1 - n * att_hat
            recon_rows.append({"N": n, "참값 함의 전이율": t_true,
                               "추정 함의 전이율": t_hat})
            log(f"{n:<24}{t_true:>22.1%}{t_hat:>22.1%}")
        log()
        n = N_STORES_IN_MARKET
        t_true, t_hat = 1 - n * att_true, 1 - n * att_hat
        log(f"가정한 N = {n}에서 읽으면:")
        log(f"  참값 ATT {att_true:.2%} → 함의 전이율 {t_true:+.1%} "
            "(거의 0 — 신규점 매출이 대부분 순증이라는 뜻)")
        log(f"  추정 ATT {att_hat:.2%} → 함의 전이율 {t_hat:+.1%} "
            "(**음수** — 자기잠식이 0인 것을 넘어 시장 자체가 커졌다는 뜻)")
        log()
        log("여기서 결정적인 것이 나온다. 추정값을 그대로 믿으면 **전이율이 음수**가 된다.")
        log("신규점이 자기 매출의 2배가 넘는 크기로 시장 전체를 키웠다는 말이다. 집적 효과로")
        log("불가능하지는 않지만, 그런 주장에는 별도의 증거가 필요하다. 근거 없이 받아들이면")
        log("자기잠식을 걱정할 필요가 없다는 결론으로 곧장 간다.")
        log()
        log("**이 검산이 분석 2의 편향을 잡아냈다.** placebo 순열추론은 p=0.027로 통과시켰다.")
        log(f"통계적 유의성은 크기의 정확성을 보증하지 않는다 — 추정치가 참값의 "
            f"{att_hat / att_true:.1f}배인데도 유의했다.")
        log("경제 논리로 되짚어 보는 검산이 통계 검정이 못 잡는 것을 잡는다. 순서가 중요하다:")
        log("추정 → 유의성 → **경제적 정합성** → 결정. 세 번째 칸을 비우면 안 된다.")
        log()
        log("[결정으로 옮기면] 추정 편향이 판정을 뒤집는 지점이 있는가")
        log("음수 전이율은 0으로 묶어 읽는다. 시장 확대를 주장할 근거가 없으므로 그 이득을")
        log("현금흐름에 태우지 않는 것이 보수적 처리다.")
        log()
        log(f"{'N':>4}{'참값 전이율':>12}{'참값 NPV':>11}{'판정':>7}"
            f"{'추정 전이율':>14}{'추정 NPV':>11}{'판정':>7}  결론")
        dec_rows = []
        flips = []
        for n_i in [10, 15, 20, 25, 30]:
            row = {"N": n_i}
            verd = {}
            for tag, a in [("참값", att_true), ("추정", att_hat)]:
                t = 1 - n_i * a
                t_eff = float(min(max(t, 0.0), 0.99))
                cf = GROSS_MONTHLY * MARGIN_RATE * (1 - t_eff)
                npv, _ = npv_and_payback(cf, CAPEX, MONTHS, r_m)
                row |= {f"{tag} 전이율": t, f"{tag} 적용 전이율": t_eff,
                        f"{tag} NPV": npv}
                verd[tag] = npv > 0
            same = verd["참값"] == verd["추정"]
            if not same:
                flips.append(n_i)
            row["판정 일치"] = same
            dec_rows.append(row)
            log(f"{n_i:>4}{row['참값 전이율']:>12.1%}{row['참값 NPV']:>11,.0f}"
                f"{('통과' if verd['참값'] else '기각'):>7}"
                f"{row['추정 전이율']:>14.1%}{row['추정 NPV']:>11,.0f}"
                f"{('통과' if verd['추정'] else '기각'):>7}"
                f"  {'같다' if same else '**뒤집힌다**'}")
        log()
        if flips:
            log(f"**N = {flips}에서 판정이 뒤집힌다.** 참값으로 계산하면 기각인 출점을 부풀린")
            log("추정치는 통과시킨다. 편향이 소수 셋째 자리의 문제가 아니라 열지 말아야 할")
            log("매장을 여는 문제로 나타난다.")
        else:
            log("이 파라미터 범위에서는 판정이 뒤집히지 않는다. 그러나 여유(NPV 크기)는 크게")
            log("다르므로, 추정 편향은 '결정'은 아니어도 '결정의 안전 여유'를 왜곡한다.")
        log()
        log("동시에 정직하게 적어 둘 것이 있다. **N도 가정값이다.** N이 얼마인지 모르면")
        log("어느 쪽 결론도 확정되지 않는다. 두 경로를 맞추는 환산 계수를 모르는 채로")
        log("시장 단위 추정치를 매장 단위 결정에 쓰는 것 자체가 위험하다. 실무에서 먼저 할")
        log("일은 더 정교한 추정이 아니라 **시장 안에 자사·경쟁 매장이 몇 개인지 세는 것**이다.")
        log()

    # =======================================================
    # (4) 용량 제약
    # =======================================================
    log("=" * 78)
    log("(4) 용량 제약 — 수요가 있어도 못 받는다")
    log("=" * 78)
    log(f"월 처리 가능 매출 상한 = {CAPACITY_MONTHLY:,.0f}만원 "
        f"(기준 매출 {GROSS_MONTHLY:,.0f}만원의 {CAPACITY_MONTHLY / GROSS_MONTHLY:.2f}배)")
    log()
    log(f"{'수요 시나리오':<20}{'수요(만원)':>12}{'실현 매출':>12}{'월 기여이익':>13}"
        f"{'NPV(만원)':>12}")
    cap_rows = []
    for label, mult in [("하방 −20%", 0.80), ("기준", 1.00),
                        ("상방 +20%", 1.20), ("상방 +40%", 1.40)]:
        demand = GROSS_MONTHLY * mult
        realized = min(demand, CAPACITY_MONTHLY)
        cf = realized * MARGIN_RATE
        npv, _ = npv_and_payback(cf, CAPEX, MONTHS, r_m)
        cap_rows.append({"시나리오": label, "수요": demand, "실현 매출": realized,
                         "월 기여이익": cf, "NPV": npv})
        log(f"{label:<20}{demand:>12,.0f}{realized:>12,.0f}{cf:>13,.0f}{npv:>12,.0f}")
    log()
    up = [r for r in cap_rows if r["시나리오"] == "상방 +40%"][0]
    base_r = [r for r in cap_rows if r["시나리오"] == "기준"][0]
    down = [r for r in cap_rows if r["시나리오"] == "하방 −20%"][0]
    log(f"상방은 잘린다: 수요가 40% 늘어도 실현 매출은 "
        f"{up['실현 매출'] / base_r['실현 매출'] - 1:+.1%}에서 멈춘다(용량 상한).")
    log(f"하방은 잘리지 않는다: 수요가 20% 줄면 매출도 그대로 20% 준다.")
    log(f"NPV로 보면 상방 +{up['NPV'] - base_r['NPV']:,.0f}만원 대 "
        f"하방 {down['NPV'] - base_r['NPV']:,.0f}만원 — **비대칭이다.**")
    log()
    log("이 비대칭이 (1)의 임계비와 같은 구조라는 점이 중요하다. 용량이 상방을 자르면")
    log("기대 이득이 줄고, 그러면 같은 예측에서도 결정이 보수적으로 밀린다. 입지 점수 상위가")
    log("곧 최선이 아닌 이유가 여기 있다 — 용량을 늘려야 하는 자리와 그대로 받을 수 있는")
    log("자리의 투자액이 다르고, 상방을 받을 수 없는 자리는 점수만큼의 값을 못 낸다.")
    log()

    # =======================================================
    # 그림·저장
    # =======================================================
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    crs = np.linspace(0.05, 0.95, 181)
    ax.plot(crs, newsvendor_q(DEMAND_MU, sigma, crs), color="#1f6f8b")
    for case, c in COST_CASES.items():
        cr = critical_ratio(c["Cu"], c["Co"])
        q = newsvendor_q(DEMAND_MU, sigma, cr)
        ax.plot([cr], [q], "o", ms=7, label=f"{case} (임계비 {cr:.2f} → {q:.0f})")
    ax.axhline(DEMAND_MU, color="gray", ls=":", lw=1.0, label="점추정 100")
    ax.set_xlabel("임계비 Cu/(Cu+Co)")
    ax.set_ylabel("최적 발주량")
    ax.set_title("(a) 같은 예측, 갈라지는 발주량")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ts = np.linspace(0.0, 0.6, 121)
    npvs = [npv_and_payback(GROSS_MONTHLY * MARGIN_RATE * (1 - t), CAPEX, MONTHS, r_m)[0]
            for t in ts]
    ax.plot(ts * 100, npvs, color="#1f6f8b")
    ax.axhline(0, color="gray", lw=1.0)
    ax.axvline(be * 100, color="crimson", ls="--", lw=1.2,
               label=f"손익분기 {be:.1%}")
    ax.axvline(25, color="#c0503d", ls=":", lw=1.2, label="실무 관행 상한 25%")
    ax.set_xlabel("전이율 (%)")
    ax.set_ylabel("NPV (만원)")
    ax.set_title("(b) 전이율과 NPV — 상한은 어디서 오는가")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "15-3-decision-economics.png", dpi=140)
    plt.close(fig)
    log("그림 저장: 15-3-decision-economics.png")

    cr_tab.to_csv(RESULTS_DIR / "15-3-critical-ratio.csv", index=False,
                  encoding="utf-8-sig")
    pd.DataFrame(npv_rows).to_csv(RESULTS_DIR / "15-3-npv.csv", index=False,
                                  encoding="utf-8-sig")
    pd.DataFrame(sens_rows).to_csv(RESULTS_DIR / "15-3-breakeven-sensitivity.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(cap_rows).to_csv(RESULTS_DIR / "15-3-capacity.csv", index=False,
                                  encoding="utf-8-sig")
    if est is not None:
        pd.DataFrame(recon_rows).to_csv(RESULTS_DIR / "15-3-reconciliation.csv",
                                        index=False, encoding="utf-8-sig")
        pd.DataFrame(dec_rows).to_csv(RESULTS_DIR / "15-3-decision.csv", index=False,
                                      encoding="utf-8-sig")
    log("표 저장: 15-3-critical-ratio.csv · 15-3-npv.csv · "
        "15-3-breakeven-sensitivity.csv · 15-3-capacity.csv · "
        "15-3-reconciliation.csv · 15-3-decision.csv")

    (RESULTS_DIR / "15-3-summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
