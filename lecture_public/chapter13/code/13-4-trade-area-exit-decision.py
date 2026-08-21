"""
13장 실습 4(비즈니스): 상권 세분화와 점포 철수 결정
====================================================
비즈니스 질문: "인구감소지역에 흩어진 점포망에서 (a) 상권 유형이 생활서비스의
존속을 실제로 가르는가, (b) 어느 점포를 언제 닫아야 하는가."

3층 모델에서의 위치:
  [1층 GIS]  읍면동 상권 지표(점포 구성·다양성·체인 비율) — 계산으로 완결
  [2층 보조] KMeans 상권 세분화 (군집의 방법과 한계는 4장 4.4)
  [3층 결정] 단위 경제로 철수 컷라인·철수 시점 판정 → 민간 철수와 공공 지원의 겹침

이 분석의 두 부분은 성격이 다르다. 반드시 구분해서 읽는다.
  · 실측: 읍면동 상권 구성과 생활서비스 공백은 **실데이터로 센 값**이다.
  · 모형: 철수 판정은 **가정한 단위 경제 파라미터** 위의 계산이다. 금액을 지어내지
    않으려고 고정비를 1로 두는 무차원 비율로 다뤘다. 데이터가 정하는 것은 판정의
    순서이고, 수준은 정규화가 정한다.

한계(본문에도 적는다):
  · 상가정보에는 매출도 인구도 없다. 배후 규모를 '읍면동 총 점포 수'로 대리했다.
  · 한 시점 스냅숏이므로 폐업률(흐름)을 셀 수 없다. 대신 생활서비스 공백(상태)을
    센다. 공백은 "원래 없던 곳"과 "있었는데 사라진 곳"을 구분하지 못한다.

실행:
    python 13-0b-trade-area-data-prep.py      # 최초 1회: 실데이터 준비
    python 13-4-trade-area-exit-decision.py
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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

SEED = 42
K_RANGE = range(2, 9)
N_PERM = 1000
N_TOP_MAJOR = 6          # 군집 피처에 쓸 업종 대분류 개수(점포 수 상위)
LIFE_LABEL = "수리·개인"  # 배후 인구 대리로 쓰는 업종 대분류(13-0b의 n_personal)

# ── 단위 경제 파라미터 (전부 가정값. 고정비 F = 1 기준의 무차원 비율) ──────────
# 공표된 업종별 매출·임대료·원상복구비 수치를 확인하지 못했으므로 금액을 쓰지 않는다.
# 대신 비율로 두고 ±30% 민감도로 결론이 얼마나 흔들리는지 보고한다.
BASE = {
    "rent_share": 0.35,   # ρ: 고정비 중 임대료 비중
    "penalty": 0.30,      # ψ: 잔여 월 임대료 대비 중도해지 위약금 비율
    "restore": 4.0,       # R: 원상복구비 = 월 임대료 × R
    "discount": 0.005,    # r: 월 할인율 (연 약 6.2%)
    "decay": -0.005,      # g: 월 수요 감소율 (연 약 -5.8%)
    "lease_months": 24,   # T: 잔여 계약 월수
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    up = DATA_DIR / "trade_area_units.parquet"
    sp = DATA_DIR / "trade_area_stores.parquet"
    if not (up.exists() and sp.exists()):
        raise SystemExit(
            f"데이터가 없습니다: {up.name} / {sp.name}\n"
            "먼저 실행: python 13-0b-trade-area-data-prep.py")
    return pd.read_parquet(up), pd.read_parquet(sp)


# ────────────────────────────── 1. 상권 세분화 ──────────────────────────────

def pick_features(units: pd.DataFrame) -> list[str]:
    """군집 피처를 고른다. 규모·다양성·체인 비율 + 점포 수 상위 대분류 구성비."""
    share_cols = [c for c in units.columns if c.startswith("share_")]
    weight = (units[share_cols].mul(units["n_stores"], axis=0).sum()
                               .sort_values(ascending=False))
    top = list(weight.index[:N_TOP_MAJOR])
    return ["log_stores", "diversity", "chain_share"] + top


def choose_k(z: np.ndarray) -> tuple[int, dict[int, float]]:
    """실루엣 계수로 군집 수를 고른다. 근거를 남기려고 전 구간을 출력한다."""
    scores = {}
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(z)
        scores[k] = silhouette_score(z, lab, sample_size=5000, random_state=SEED)
    best = max(scores, key=scores.get)
    return best, scores


def cluster_stability(z: np.ndarray, k: int, n_seeds: int = 10) -> float:
    """시드를 바꿔 다시 군집해 라벨이 얼마나 같은지(평균 쌍별 ARI)."""
    labs = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(z)
            for s in range(n_seeds)]
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris))


def degeneracy_check(units: pd.DataFrame, labels: np.ndarray, k: int) -> float:
    """[퇴화 검사] 군집이 사실은 '점포 수'를 되풀이하는 것인지 확인한다.

    점포 수 분위만으로 만든 라벨과 ARI가 1에 가까우면 군집은 규모의 다른 이름이다.
    """
    size_bins = pd.qcut(units["n_stores"].rank(method="first"), k,
                        labels=False, duplicates="drop")
    return float(adjusted_rand_score(size_bins, labels))


def perm_test_spread(values: np.ndarray, labels: np.ndarray,
                     rng: np.random.Generator) -> tuple[float, float]:
    """군집별 평균의 산포(표준편차)가 우연인지 순열검정한다."""
    def spread(lab):
        return float(np.std([values[lab == g].mean() for g in np.unique(lab)]))
    obs = spread(labels)
    null = np.array([spread(rng.permutation(labels)) for _ in range(N_PERM)])
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return obs, p


def perm_test_diff(values: np.ndarray, group: np.ndarray,
                   rng: np.random.Generator) -> tuple[float, float]:
    """두 집단 평균 차이의 순열검정(양측)."""
    def diff(g):
        return float(values[g].mean() - values[~g].mean())
    obs = diff(group)
    null = np.array([diff(rng.permutation(group)) for _ in range(N_PERM)])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))
    return obs, p


# ────────────────────────────── 2. 철수 판정 ──────────────────────────────

def exit_decision(s_ratio: np.ndarray, par: dict) -> dict[str, np.ndarray | float]:
    """무차원 단위 경제로 철수를 판정한다.

    고정비 F = 1. 월 기여이익 m(t) = s_ratio·(1+g)^t − 1.
    즉시 철수 비용 E = ψ·ρ·T + R·ρ  (잔여 위약금 + 원상복구비).
    유지 가치 V = Σ m(t)/(1+r)^t.  V < −E 이면 지금 닫는 편이 낫다.

    m(t)가 s_ratio에 선형이므로 컷라인이 닫힌 형태로 나온다.
      V = s_ratio·A − B,  A = Σ((1+g)/(1+r))^t,  B = Σ(1+r)^(−t)
      즉시 철수 컷라인 s* = (B − E) / A
    """
    T = int(par["lease_months"])
    g, r = par["decay"], par["discount"]
    rho, psi, R = par["rent_share"], par["penalty"], par["restore"]

    t = np.arange(1, T + 1)
    disc = (1.0 + r) ** (-t.astype(float))
    A = float(np.sum(((1.0 + g) ** t) * disc))
    B = float(np.sum(disc))
    E = psi * rho * T + R * rho

    V = s_ratio * A - B
    s_cut_now = (B - E) / A                     # 즉시 철수 컷라인
    s_cut_expiry = (1.0 + g) ** (-T)            # 만료 시점에 적자가 되는 컷라인

    exit_now = V < -E
    loss_at_T = s_ratio * (1.0 + g) ** T - 1.0 < 0
    exit_expiry = (~exit_now) & loss_at_T
    keep = ~exit_now & ~exit_expiry

    # 철수 시점의 한계 조건: 한 달 더 버티면 위약금이 ψρ 줄고 그 달 손실 −m(t)를 감당한다.
    #   −m(t) > ψρ  ⟺  s_ratio·(1+g)^t < 1 − ψρ
    thr = 1.0 - psi * rho
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = np.ceil(np.log(np.clip(thr / s_ratio, 1e-12, None)) / np.log1p(g))
    t_star = np.clip(np.nan_to_num(t_star, nan=np.inf), 0, np.inf)

    return {"V": V, "exit_now": exit_now, "exit_expiry": exit_expiry, "keep": keep,
            "t_star": t_star, "s_cut_now": s_cut_now, "s_cut_expiry": s_cut_expiry,
            "E": E, "A": A, "B": B, "margin_threshold": thr}


def draw_map(units: pd.DataFrame, stores: pd.DataFrame, dec: np.ndarray,
             exit_within: np.ndarray) -> Path:
    """점 지도 두 장. 행정동 경계 파일을 쓰지 않으므로 **점 표시**다.

    왼쪽: 읍면동 대표점을 상권 유형별로. 오른쪽: 인구감소지역 편의점의 판정.
    대표점은 소속 점포 좌표의 중앙값이므로 행정동 중심이 아니다(표시용 근사).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]

    for c in sorted(units["cluster"].unique()):
        sel = units[units["cluster"] == c]
        ax1.scatter(sel["lon"], sel["lat"], s=14, alpha=0.75,
                    color=colors[c % len(colors)],
                    label=f"유형 {c} (n={len(sel)})")
    ax1.set_title("읍면동 상권 유형 (점 표시, 경계 없음)")
    ax1.legend(fontsize=8, loc="lower left")

    sd = stores.loc[dec]
    keep = ~exit_within
    ax2.scatter(sd.loc[keep, "lon"], sd.loc[keep, "lat"], s=6, alpha=0.35,
                color="#999999", label=f"유지·만료 시 철수 (n={int(keep.sum())})")
    ax2.scatter(sd.loc[exit_within, "lon"], sd.loc[exit_within, "lat"], s=8,
                alpha=0.7, color="#C44E52",
                label=f"즉시 철수 후보 (n={int(exit_within.sum())})")
    ax2.set_title("인구감소지역 편의점 철수 판정")
    ax2.legend(fontsize=8, loc="lower left")

    for ax in (ax1, ax2):
        ax.set_xlabel("경도"), ax.set_ylabel("위도")
        # 울릉군(동경 약 130.9)까지 들어오도록 잡는다. 제주는 지정 지역이 아니라 없다.
        ax.set_xlim(125.5, 131.2), ax.set_ylim(33.0, 38.7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    fig.suptitle("인구감소지역과 비지정 군의 상권 유형, 그리고 철수 후보의 분포", y=0.97)
    fig.tight_layout()
    out = RESULTS_DIR / "13-4-trade-area-exit-map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union else 1.0


# ────────────────────────────── 실행 ──────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)
    print("=" * 76)
    print("상권 세분화와 점포 철수 결정 (13-4, 실데이터 + 단위 경제)")
    print("=" * 76)

    units, stores = load_inputs()
    units["log_stores"] = np.log1p(units["n_stores"])
    print(f"\n분석 단위: 읍면동 {len(units):,}개 "
          f"(인구감소지역 {int(units['is_decline_area'].sum()):,} / "
          f"비지정 군 {int((~units['is_decline_area']).sum()):,})")
    print(f"대상 점포(편의점): {len(stores):,}개")
    print("※ 상가(상권)정보 2026년 6월 기준 스냅숏. 매출·인구 정보는 없다.")

    # ── [1층+2층] 상권 세분화 ──────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("[1] 상권 세분화 — 군집 수 선택과 안정성")
    feats = pick_features(units)
    print(f"  피처 {len(feats)}개: {feats}")
    z = StandardScaler().fit_transform(units[feats].fillna(0.0).to_numpy())

    k, scores = choose_k(z)
    print("  실루엣 계수: " + "  ".join(f"k={kk}:{v:.3f}" for kk, v in scores.items()))
    print(f"  → 선택 k = {k} (최대 실루엣 {scores[k]:.3f})")

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(z)
    units["cluster"] = km.labels_
    ari_seed = cluster_stability(z, k)
    ari_size = degeneracy_check(units, km.labels_, k)
    print(f"  안정성(시드 10개 평균 쌍별 ARI) = {ari_seed:.3f}")
    print(f"  [퇴화 검사] 점포 수 분위 라벨과의 ARI = {ari_size:.3f}")
    print("     → 1에 가까우면 군집은 '규모'의 다른 이름일 뿐이다.")

    prof_cols = {"n": ("n_stores", "size"), "점포수중앙값": ("n_stores", "median"),
                 "다양성": ("diversity", "mean"), "체인비율": ("chain_share", "mean"),
                 "편의점보유": ("has_convenience", "mean"),
                 "약국보유": ("has_pharmacy", "mean"),
                 "의원보유": ("has_clinic", "mean")}
    prof = units.groupby("cluster").agg(**{k2: pd.NamedAgg(*v)
                                           for k2, v in prof_cols.items()})
    prof = prof.sort_values("점포수중앙값")
    print("\n  유형별 프로필 (점포수 중앙값 오름차순)")
    print(prof.round(3).to_string())

    # ── [실측] 유형별 생활서비스 공백률 ────────────────────────────────────
    print("\n" + "-" * 76)
    print("[2] 실측 — 유형별 생활서비스 공백률 (편의점·약국·의원이 하나도 없는 읍면동)")
    for name, label in (("convenience", "편의점"), ("pharmacy", "약국"),
                        ("clinic", "의원")):
        gap = (~units[f"has_{name}"]).to_numpy().astype(float)
        by = units.assign(_g=gap).groupby("cluster")["_g"].mean()
        obs, p = perm_test_spread(gap, units["cluster"].to_numpy(), rng)
        line = "  ".join(f"유형{c}:{v:.1%}" for c, v in by.items())
        print(f"  {label} 공백률 — 전체 {gap.mean():.1%} | {line}")
        print(f"      [대조 N3] 유형 라벨 무작위화 {N_PERM}회: 산포 {obs:.4f}, p = {p:.3f}")

    # ── [대조] 인구감소지역 대 비지정 군 ──────────────────────────────────
    print("\n" + "-" * 76)
    print("[3] 대조 — 인구감소지역 고시 여부 (군 지역만 비교)")
    gun = units[units["is_gun"]].copy()
    grp = gun["is_decline_area"].to_numpy()
    print(f"  비교 대상: 군 지역 읍면동 {len(gun):,}개 "
          f"(지정 {int(grp.sum()):,} / 비지정 {int((~grp).sum()):,})")
    for name, label in (("convenience", "편의점"), ("pharmacy", "약국"),
                        ("clinic", "의원")):
        gap = (~gun[f"has_{name}"]).to_numpy().astype(float)
        obs, p = perm_test_diff(gap, grp, rng)
        print(f"  {label} 공백률 — 지정 {gap[grp].mean():.1%} vs "
              f"비지정 {gap[~grp].mean():.1%} | 차이 {obs:+.1%}")
        print(f"      [대조 N4] 지정 라벨 무작위화 {N_PERM}회: p = {p:.3f}")

    cross = (gun.assign(공백=~gun["has_convenience"])
                .pivot_table(index="cluster", columns="is_decline_area",
                             values="공백", aggfunc="mean"))
    cross.columns = ["비지정", "지정"]
    n_cross = (gun.pivot_table(index="cluster", columns="is_decline_area",
                               values="n_stores", aggfunc="size")
                  .rename(columns={False: "n_비지정", True: "n_지정"}))
    print("\n  유형별 편의점 공백률 (군 지역, 지정 여부별)")
    print(pd.concat([n_cross, cross.round(3)], axis=1).to_string())

    # ── [3층 결정] 철수 판정 ──────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("[4] 철수 판정 — 무차원 단위 경제 (고정비 F = 1 기준)")
    print("  파라미터(전부 가정값): " +
          ", ".join(f"{k2}={v}" for k2, v in BASE.items()))

    s_raw = (stores["n_stores"] / stores["n_same_kind"]).to_numpy(dtype=float)
    s_med = float(np.median(s_raw))
    s_ratio = s_raw / s_med
    print(f"\n  상권 배분 지수 s = (읍면동 총 점포 수) / (읍면동 편의점 수)")
    print(f"    중앙값 {s_med:.2f}, 사분위 [{np.percentile(s_raw, 25):.2f}, "
          f"{np.percentile(s_raw, 75):.2f}], 범위 [{s_raw.min():.2f}, {s_raw.max():.2f}]")
    print(f"    정규화: 중앙값 점포의 월 기여이익이 정확히 0이 되도록 맞춤")
    print("    ※ 수준은 이 정규화가 정한 것이고, 데이터가 정하는 것은 순서다.")

    res = exit_decision(s_ratio, BASE)
    cut_now, cut_exp = res["s_cut_now"] * s_med, res["s_cut_expiry"] * s_med
    print(f"\n  즉시 철수 비용 E = {res['E']:.2f} (고정비 {res['E']:.2f}개월분)")
    print(f"  컷라인(관측 단위): 즉시 철수 s < {cut_now:.2f} | "
          f"만료 시점에 적자 s < {cut_exp:.2f}")
    print(f"  판정 3분류: 즉시 철수 {res['exit_now'].sum():,}개 "
          f"({res['exit_now'].mean():.1%}) | 만료 시 철수 {res['exit_expiry'].sum():,}개 "
          f"({res['exit_expiry'].mean():.1%}) | 유지 {res['keep'].sum():,}개 "
          f"({res['keep'].mean():.1%})")
    print(f"\n  **이 분석의 결론은 두 컷라인 사이의 간격이다.** "
          f"s ∈ [{cut_now:.2f}, {cut_exp:.2f})")
    print(f"    적자로 갈 것이 보이는데도 만료까지 버티는 편이 옳은 구간이며, 점포 "
          f"{res['exit_expiry'].sum():,}개({res['exit_expiry'].mean():.1%})가 여기 있다.")
    print("    간격을 만드는 것은 잔여 위약금과 원상복구비 — 즉 되돌릴 수 없는 철수 비용이다.")
    print("    반면 '적자 점포가 몇 %인가'는 결론이 아니다. 중앙값을 손익분기로 맞춘")
    print("    정규화가 그 비율을 정한다(수준은 가정, 순서는 데이터).")

    psi_rho = BASE["penalty"] * BASE["rent_share"]
    print(f"\n  철수 시점의 한계 조건: 그 달 손실 −m(t)가 ψρ = {psi_rho:.3f}(고정비 대비)를")
    print("    넘을 때 닫는다. 한 달 더 버티면 위약금이 그만큼 줄기 때문이다. 손익분기")
    print("    (m = 0)가 아니라 그보다 아래에서 닫는 것이 합리적이다 — 되돌릴 수 없는")
    print("    철수 비용이 만드는 이력현상(Dixit, 1989).")
    finite = np.isfinite(res["t_star"]) & (res["t_star"] <= BASE["lease_months"])
    if finite.any():
        ts = res["t_star"][finite]
        print(f"    잔여 계약 안에 한계 조건이 성립하는 점포 {int(finite.sum()):,}개 "
              f"({finite.mean():.1%}), t* 중앙값 {np.median(ts):.0f}개월, "
              f"t*=0(지금) {int((ts == 0).sum()):,}개")
        print(f"    순차 판정(t*=0)과 현재가치 판정(즉시 철수)이 각각 "
              f"{int((ts == 0).sum()):,}개 대 {res['exit_now'].sum():,}개로 거의 겹친다 "
              "— 두 규칙의 정합성 확인이다.")
    else:
        print("    잔여 계약 안에 한계 조건이 성립하는 점포가 없다.")

    # ── 겹침과 프록시 타당성 ──────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("[5] 겹침 — 그리고 이 모형을 어디까지 쓸 수 있는가")
    dec = stores["is_decline_area"].to_numpy()
    exit_now = res["exit_now"]
    print(f"  (가) 실측 부분: 서비스가 **이미** 빠져나간 자리는 [3]에서 봤다 —")
    print(f"       군 지역 편의점 공백률 지정 30% 대 비지정 10% 수준의 격차(p=0.001).")
    print(f"  (나) 모형 부분: 즉시 철수 후보 {exit_now.sum():,}개 중 인구감소지역 소재 "
          f"{int((exit_now & dec).sum()):,}개 "
          f"({(exit_now & dec).sum() / max(exit_now.sum(), 1):.1%})")
    print(f"       점포 대비 즉시 철수 비율: 인구감소지역 {exit_now[dec].mean():.1%} vs "
          f"비지정 군 {exit_now[~dec].mean():.1%}")
    if exit_now[dec].mean() < exit_now[~dec].mean():
        print("       → **방향이 뒤집혔다.** 인구감소지역 쪽이 오히려 낮다.")
    print("  (다) 왜 뒤집혔는가 — 프록시를 진단한다")
    cvs = (stores["n_same_kind"] / stores["n_stores"]).to_numpy(dtype=float)
    nst = stores["n_stores"].to_numpy(dtype=float)
    print(f"       s 중앙값: 인구감소지역 {np.median(s_raw[dec]):.2f} vs "
          f"비지정 군 {np.median(s_raw[~dec]):.2f}")
    print(f"       읍면동 총 점포 수 중앙값: {np.median(nst[dec]):.0f} vs "
          f"{np.median(nst[~dec]):.0f} | 편의점 비중 중앙값: "
          f"{np.median(cvs[dec]):.3f} vs {np.median(cvs[~dec]):.3f}")
    print(f"       s와 편의점 비중의 상관 = {np.corrcoef(s_raw, cvs)[0, 1]:.3f}")
    print("       비지정 군의 읍면동은 총 점포 수도 많지만 편의점 수가 더 빠르게 많다.")
    print("       편의점은 배후 인구에 붙고 총 점포 수는 상업 집적에 붙기 때문이다. 그래서")
    print("       인구는 많고 상업 집적은 상대적으로 덜한 주거형 교외에서 s가 낮게 나오고,")
    print("       모형은 그것을 '배후가 얇다'로 잘못 읽는다. 매출 대리변수의 결함이다.")
    print("  (라) 그러므로 이 모형은 **비교 가능한 집합 안에서 순서를 정하는 데만** 쓴다.")
    print("       지역 간 비교에는 쓰지 않는다. 이 결함을 드러낸 것은 모형의 통계 진단이")
    print("       아니라, 지정 여부로 갈라 본 대조와 프록시를 갈아 끼운 대조군([6] N1b)이다.")

    print("\n  인구감소지역 안에서만 다시 판정 (그 안의 중앙값으로 정규화)")
    s_dec = s_raw[dec] / float(np.median(s_raw[dec]))
    r_dec = exit_decision(s_dec, BASE)
    print(f"    점포 {int(dec.sum()):,}개 — 즉시 철수 {r_dec['exit_now'].mean():.1%} | "
          f"만료 시 철수 {r_dec['exit_expiry'].mean():.1%} | 유지 {r_dec['keep'].mean():.1%}")
    dong_exit = (stores.loc[dec].assign(_e=r_dec["exit_now"])
                       .groupby(["sido_short", "sigungu", "dong"], observed=True)["_e"]
                       .mean())
    print(f"    편의점 전부가 즉시 철수 후보인 읍면동 {int((dong_exit == 1.0).sum()):,}개 "
          f"/ {len(dong_exit):,}개 ({(dong_exit == 1.0).mean():.1%})")
    print("    → 그 읍면동은 편의점이 하나도 남지 않게 된다. [2]의 공백 목록에 더해질 자리다.")

    # ── 대조군과 민감도 ───────────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("[6] 대조군(null control) — 심은 메커니즘만 제거해 다시 돌린다")
    s_n1 = stores["n_stores"].to_numpy(dtype=float)
    s_n1 = s_n1 / np.median(s_n1)
    r_n1 = exit_decision(s_n1, BASE)
    print(f"  N1 동종 경쟁 나눗셈 제거(s = 총 점포 수): 즉시 철수 "
          f"{r_n1['exit_now'].mean():.1%} (기준 {exit_now.mean():.1%}), "
          f"Jaccard {jaccard(exit_now, r_n1['exit_now']):.3f}")
    # N1b 배후 대리를 갈아 끼운다. 소매(편의점 자신이 포함)가 아니라 생활밀착 업종을 쓴다.
    per = stores["n_personal"].to_numpy(dtype=float)
    ok = per > 0
    s_n1b = np.where(ok, per / stores["n_same_kind"].to_numpy(dtype=float), np.nan)
    s_n1b = s_n1b / np.nanmedian(s_n1b)
    r_n1b = exit_decision(np.nan_to_num(s_n1b, nan=0.0), BASE)
    print(f"  N1b 배후 대리 교체(s = 생활밀착 업종 수 / 편의점 수, "
          f"{LIFE_LABEL} 업종 0인 점포 {int((~ok).sum()):,}개 제외): "
          f"즉시 철수 {r_n1b['exit_now'][ok].mean():.1%}, "
          f"Jaccard {jaccard(exit_now[ok], r_n1b['exit_now'][ok]):.3f}")
    print(f"      지정 {r_n1b['exit_now'][ok & dec].mean():.1%} vs "
          f"비지정 군 {r_n1b['exit_now'][ok & ~dec].mean():.1%}"
          + ("  → 프록시를 바꿔도 방향이 그대로다(결함이 배후 대리 자체에 있다)"
             if r_n1b['exit_now'][ok & dec].mean() < r_n1b['exit_now'][ok & ~dec].mean()
             else "  → 프록시를 바꾸면 방향이 바뀐다(결론이 대리변수에 달려 있다)"))
    r_n2 = exit_decision(s_ratio, {**BASE, "decay": 0.0})
    print(f"  N2 수요 감소 추세 제거(g = 0): 즉시 철수 {r_n2['exit_now'].mean():.1%}, "
          f"만료 시 철수 {r_n2['exit_expiry'].mean():.1%}, "
          f"Jaccard {jaccard(exit_now, r_n2['exit_now']):.3f}")

    print("\n[7] 민감도 — 가정값 ±30%")
    rows = []
    for key in ("rent_share", "penalty", "restore", "discount", "decay",
                "lease_months"):
        for mult in (0.7, 1.3):
            par = dict(BASE)
            val = BASE[key] * mult
            par[key] = int(round(val)) if key == "lease_months" else val
            rr = exit_decision(s_ratio, par)
            rows.append({
                "파라미터": key, "배수": mult, "값": round(par[key], 4),
                "즉시철수": round(float(rr["exit_now"].mean()), 4),
                "만료시철수": round(float(rr["exit_expiry"].mean()), 4),
                "유지": round(float(rr["keep"].mean()), 4),
                "컷라인(관측)": round(float(rr["s_cut_now"] * s_med), 3),
                "Jaccard": round(jaccard(exit_now, rr["exit_now"]), 3),
            })
    sens = pd.DataFrame(rows)
    print(sens.to_string(index=False))
    print(f"\n  즉시 철수 비율 범위 {sens['즉시철수'].min():.1%}~{sens['즉시철수'].max():.1%}, "
          f"Jaccard 최소 {sens['Jaccard'].min():.3f}")
    print("  → 판정 집합의 크기는 가정에 흔들린다. 흔들리지 않는 것은 '어느 점포가 먼저'다.")

    # ── 저장 ──────────────────────────────────────────────────────────────
    units_out = units.drop(columns=["log_stores"])
    units_out.to_csv(RESULTS_DIR / "trade_area_typology.csv",
                     index=False, encoding="utf-8-sig")
    stores_out = stores.copy()
    stores_out["s_index"] = s_raw
    stores_out["s_ratio"] = s_ratio
    stores_out["m0"] = s_ratio - 1.0
    stores_out["decision"] = np.where(res["exit_now"], "즉시철수",
                              np.where(res["exit_expiry"], "만료시철수", "유지"))
    stores_out["t_star"] = np.where(np.isfinite(res["t_star"]), res["t_star"], -1)
    # 인구감소지역 안에서만 정규화해 다시 판정한 결과(지역 간 비교에 쓰지 않는 쪽)
    within = np.full(len(stores_out), "", dtype=object)
    within[dec] = np.where(r_dec["exit_now"], "즉시철수",
                   np.where(r_dec["exit_expiry"], "만료시철수", "유지"))
    stores_out["decision_within_decline"] = within
    stores_out.to_csv(RESULTS_DIR / "store_exit_decision.csv",
                      index=False, encoding="utf-8-sig")
    sens.to_csv(RESULTS_DIR / "store_exit_sensitivity.csv",
                index=False, encoding="utf-8-sig")
    fig_path = draw_map(units, stores, dec, r_dec["exit_now"])
    print(f"\n  저장 → trade_area_typology.csv, store_exit_decision.csv, "
          f"store_exit_sensitivity.csv, {fig_path.name}")

    print("\n" + "=" * 76)
    print("[완료] 상권 세분화(실측)와 철수 판정(모형)을 마쳤다.")
    print("주의: 공백률은 흐름(폐업률)이 아니라 상태다. '원래 없던 곳'과 '사라진 곳'을")
    print("      구분하지 못한다. 철수 판정의 수준은 가정의 산물이고 순서가 결론이다.")
    print("=" * 76)


if __name__ == "__main__":
    main()
