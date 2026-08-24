"""
7장 실습 3: 매출 예측과 발주량 결정 (LSTM + 분할 conformal → 임계비)
=====================================================================
비즈니스 질문: "내일 몇 개를 발주할 것인가?"

7-2는 예측을 구간으로 바꾸는 데까지 갔다. 구간은 "완충을 마련하라"까지만
말하고 "얼마나"는 말하지 않는다. 그 얼마를 정하는 것은 예측 모델이 아니라
결품 손실 Cu와 잉여 손실 Co의 비율이다. 임계비 CR = Cu/(Cu+Co)가 목표
서비스 수준이 되고, 최적 발주량은 수요 분포의 CR 분위수다(표준 결과).

이 코드가 하는 일
-----------------
1. 점포×품목×일 수요 패널을 **시간순 3분할**(학습 60 / 보정 20 / 시험 20)
2. 품목별 LSTM 점추정(과거 28일 + 내일의 알려진 캘린더 정보 → 내일 수요)
3. **정규화 분할 conformal**: 학습 잔차로 점포별 오차 규모 σ̂를 적합하고,
   **보정 집합**의 정규화 잔차 분위수로 임의 분위 τ의 예측값을 만든다.
   학습·보정·시험이 분리돼 있으므로 시험 포함률에 낙관 편향이 없다.
4. 임계비로 발주 분위를 정하고 세 정책의 실현 손익을 시험 집합에서 채점
5. 대조군 C1(등분산 데이터)·C2(비정규화 conformal)
6. 비용비를 틀리는 대가와 예측을 개선하는 이득의 크기 비교

데이터: 7-0b가 저장한 합성 패널(참값을 알아야 발주 정책을 채점할 수 있다).
비용 파라미터는 전부 **가정값**이며, 결론은 금액이 아니라 임계비의 방향과
그로 인한 분위 선택의 역전에 둔다.

실행:
    python 7-0b-demand-simdata.py        # 최초 1회: 데이터 준비
    python 7-3-demand-newsvendor.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
# 대조군 C2는 구간 폭이 상수라 순위상관이 정의되지 않는다 — 의도한 결과이므로 경고를 끈다
warnings.filterwarnings("ignore", message="An input array is constant")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
WINDOW = 28                     # 과거 28일
TRAIN_FRAC, CALIB_FRAC = 0.60, 0.20
EPOCHS, BATCH, LR = 40, 256, 0.01
TARGET_COVER = 0.90             # 90% 예측구간

# ---------------------------------------------------------------
# 비용 가정값 — 전부 가정이다. 실제 업종 원가를 인용할 근거가 없다.
# 두 품목의 손실 구조가 정반대라는 점만이 이 예제의 논지다.
# ---------------------------------------------------------------
COST = {
    # 도시락: 당일 폐기, 잔존가치 0 → 남는 손해(Co)가 모자란 손해(Cu)보다 크다
    "도시락": dict(Cu=1200, Co=2000, 설명="Cu=판매 마진, Co=원가+폐기비"),
    # 우산: 못 팔면 기회를 잃고, 남으면 다음 시즌에 팔린다 → Cu가 훨씬 크다
    "우산": dict(Cu=15000, Co=1500, 설명="Cu=마진+기회 상실, Co=재고 유지비"),
}


def get_device():
    """CUDA → MPS → CPU 자동 폴백."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 가속기를 쓰면 LSTM 커널이 달라져 같은 시드에서도 소수점 아래가 어긋난다.
# 이 실습은 CPU로 20초면 끝나는 소형 모델이므로, 누가 어디서 돌려도 같은 수치가
# 나오도록 CPU로 고정한다(7-2와 같은 판단). 큰 모델을 돌릴 때는 PIN_CPU를 끈다.
PIN_CPU = True
DEVICE = torch.device("cpu") if PIN_CPU else get_device()


# ===============================================================
# 1. 데이터 → 창(window)
# ===============================================================
def load_panel(name):
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(f"데이터가 없습니다: {path}\n먼저 실행: python 7-0b-demand-simdata.py")
    return pd.read_parquet(path)


def make_windows(df_item, n_train_day):
    """점포별 시계열을 (과거 28일, 내일의 캘린더 정보) → 내일 수요 창으로 변환.

    표준화 통계는 **학습 구간만** 써서 계산한다(미래 정보 누수 차단).
    내일의 프로모션·요일·공휴일은 발주 시점에 이미 아는 값이므로 입력에 넣는다.
    """
    stores = sorted(df_item["store_id"].unique())
    X_seq, X_cov, y, meta = [], [], [], []
    stats = {}
    for si, sid in enumerate(stores):
        d = df_item[df_item["store_id"] == sid].sort_values("day")
        dem = d["demand"].to_numpy(dtype=np.float32)
        promo = d["promo"].to_numpy(dtype=np.float32)
        hol = d["holiday"].to_numpy(dtype=np.float32)
        dow = d["dow"].to_numpy()
        mu, sd = dem[:n_train_day].mean(), dem[:n_train_day].std()
        stats[sid] = (float(mu), float(sd))
        norm = (dem - mu) / sd
        for t in range(WINDOW, len(dem)):
            X_seq.append(norm[t - WINDOW:t])
            X_cov.append([promo[t], hol[t],
                          np.sin(2 * np.pi * dow[t] / 7), np.cos(2 * np.pi * dow[t] / 7)])
            y.append(norm[t])
            meta.append((si, sid, t, dow[t], dem[t]))
    X_seq = np.array(X_seq, dtype=np.float32)[:, :, None]
    X_cov = np.array(X_cov, dtype=np.float32)
    y = np.array(y, dtype=np.float32)[:, None]
    meta = pd.DataFrame(meta, columns=["store_idx", "store_id", "day", "dow", "actual"])
    return X_seq, X_cov, y, meta, stats, stores


class DemandLSTM(nn.Module):
    """28일 창을 LSTM으로 요약하고, 내일의 알려진 캘린더 정보를 붙여 예측한다."""

    def __init__(self, hidden=32, n_cov=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden + n_cov, 1)

    def forward(self, seq, cov):
        out, _ = self.lstm(seq)
        h = self.drop(out[:, -1, :])
        return self.fc(torch.cat([h, cov], dim=1))


def train_lstm(Xs, Xc, y, seed=SEED):
    torch.manual_seed(seed)
    model = DemandLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Xs_t = torch.tensor(Xs, device=DEVICE)
    Xc_t = torch.tensor(Xc, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    n = len(Xs)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xs_t[idx], Xc_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    return model, float(loss.item())


def predict(model, Xs, Xc):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xs, device=DEVICE), torch.tensor(Xc, device=DEVICE))
    return out.squeeze(-1).cpu().numpy()


# ===============================================================
# 2. 정규화 분할 conformal
# ===============================================================
def fit_scale(pred, resid_abs, store_idx, n_stores, normalized=True):
    """점포별 오차 규모 σ̂를 학습 잔차에 적합한다.

    σ̂(s, ŷ) = a_s + b_s·ŷ — "이 점포에서 이 수준의 수요를 예측할 때
    평균적으로 얼마나 빗나갔는가"를 학습 구간의 **실제 잔차**로 잰 값이다.
    예측값의 함수로 불확실성을 '정의'하는 것이 아니라, 측정된 오차를 적합한다.

    normalized=False면 대조군 C2 — 모든 예측에 같은 폭(전역 평균 절대잔차).
    """
    floor = 0.25 * resid_abs.mean()
    if not normalized:
        return np.full(len(pred), resid_abs.mean()), None
    coefs = {}
    for s in range(n_stores):
        m = store_idx == s
        A = np.column_stack([np.ones(m.sum()), pred[m]])
        coef, *_ = np.linalg.lstsq(A, resid_abs[m], rcond=None)
        coefs[s] = coef

    def apply(p, si):
        out = np.empty(len(p))
        for s in range(n_stores):
            m = si == s
            out[m] = coefs[s][0] + coefs[s][1] * p[m]
        return np.maximum(out, floor)

    return apply(pred, store_idx), (coefs, floor, apply)


def conformal_quantile(scores, tau):
    """분할 conformal의 유한표본 보정 분위수.

    tau ≥ 0.5면 위쪽으로 보수적인 순위(ceil), tau < 0.5면 아래쪽으로 보수적인
    순위(floor)를 골라 목표 확률을 밑돌지 않게 한다.
    """
    s = np.sort(scores)
    n = len(s)
    if tau >= 0.5:
        k = int(np.ceil((n + 1) * tau))
    else:
        k = int(np.floor((n + 1) * tau))
    k = int(np.clip(k, 1, n))
    return float(s[k - 1])


# ===============================================================
# 3. 파이프라인 1회 = 한 품목 × 한 데이터셋
# ===============================================================
def run_item(df, item, label, verbose=True):
    d = df[df["item"] == item].copy()
    n_days = d["day"].nunique()
    n_train_day = int(n_days * TRAIN_FRAC)
    n_calib_day = int(n_days * (TRAIN_FRAC + CALIB_FRAC))

    Xs, Xc, y, meta, stats, stores = make_windows(d, n_train_day)
    n_stores = len(stores)
    tr = (meta["day"] < n_train_day).to_numpy()
    ca = ((meta["day"] >= n_train_day) & (meta["day"] < n_calib_day)).to_numpy()
    te = (meta["day"] >= n_calib_day).to_numpy()

    model, final_loss = train_lstm(Xs[tr], Xc[tr], y[tr])

    # 예측을 원 단위로 되돌린다(점포별 표준화의 역변환)
    def denorm(pred_norm, mask):
        out = np.empty(mask.sum())
        sub = meta[mask]
        for i, sid in enumerate(stores):
            m = (sub["store_id"] == sid).to_numpy()
            mu, sd = stats[sid]
            out[m] = pred_norm[m] * sd + mu
        return out

    pred_tr = denorm(predict(model, Xs[tr], Xc[tr]), tr)
    pred_ca = denorm(predict(model, Xs[ca], Xc[ca]), ca)
    pred_te = denorm(predict(model, Xs[te], Xc[te]), te)
    act_tr = meta.loc[tr, "actual"].to_numpy()
    act_ca = meta.loc[ca, "actual"].to_numpy()
    act_te = meta.loc[te, "actual"].to_numpy()
    si_tr = meta.loc[tr, "store_idx"].to_numpy()
    si_ca = meta.loc[ca, "store_idx"].to_numpy()
    si_te = meta.loc[te, "store_idx"].to_numpy()

    rmse_te = float(np.sqrt(np.mean((pred_te - act_te) ** 2)))

    res = dict(item=item, label=label, n_train=int(tr.sum()), n_calib=int(ca.sum()),
               n_test=int(te.sum()), rmse=rmse_te, final_train_mse=final_loss,
               meta_test=meta[te].reset_index(drop=True), pred_test=pred_te,
               act_test=act_te, si_test=si_te, stores=stores)

    # --- 정규화 / 비정규화 두 conformal ---
    for norm_flag, key in ((True, "norm"), (False, "plain")):
        sig_tr, fitted = fit_scale(pred_tr, np.abs(act_tr - pred_tr), si_tr,
                                   n_stores, normalized=norm_flag)
        if norm_flag:
            apply = fitted[2]
            sig_ca, sig_te = apply(pred_ca, si_ca), apply(pred_te, si_te)
        else:
            const = sig_tr[0]
            sig_ca = np.full(len(pred_ca), const)
            sig_te = np.full(len(pred_te), const)
        e_ca = (act_ca - pred_ca) / sig_ca            # 부호 있는 정규화 잔차

        def q_of(tau, _e=e_ca, _p=pred_te, _s=sig_te):
            return _p + conformal_quantile(_e, tau) * _s

        lo, hi = q_of((1 - TARGET_COVER) / 2), q_of(1 - (1 - TARGET_COVER) / 2)
        inside = (act_te >= lo) & (act_te <= hi)
        cover = float(np.mean(inside))
        halfwidth = (hi - lo) / 2
        # 비정규화는 폭이 상수라 순위상관이 정의되지 않는다(nan) — 그것이 C2의 요점이다
        rho = float(pd.Series(halfwidth).corr(pd.Series(np.abs(act_te - pred_te)),
                                              method="spearman"))
        # 점포별 조건부 포함률 — 주변 포함률이 맞아도 점포별로는 어긋날 수 있다
        by_store = {stores[s]: float(np.mean(inside[si_te == s])) for s in range(n_stores)}
        # 수요 수준별(예측값 3분위) 조건부 포함률 — 이분산에서 상수 폭이 깨지는 자리
        edges = np.quantile(pred_te, [1 / 3, 2 / 3])
        tier = np.digitize(pred_te, edges)
        by_tier = [float(np.mean(inside[tier == g])) for g in range(3)]
        res[key] = dict(q_of=q_of, cover=cover, width=float(np.mean(hi - lo)),
                        halfwidth=halfwidth, rho=rho, by_store=by_store,
                        by_tier=by_tier, tier_spread=float(max(by_tier) - min(by_tier)),
                        lo=lo, hi=hi, sig_te=sig_te,
                        width_cv=float(np.std(halfwidth) / np.mean(halfwidth)))

    # 교환가능성 점검 — 보정 구간과 시험 구간에서 잔차의 중심이 옮겨 갔는가
    res["drift"] = dict(bias_calib=float(np.mean(act_ca - pred_ca)),
                        bias_test=float(np.mean(act_te - pred_te)),
                        mean_calib=float(act_ca.mean()), mean_test=float(act_te.mean()))

    if verbose:
        print(f"\n  [{label} · {item}] 창 {len(Xs):,}개 "
              f"(학습 {tr.sum():,} / 보정 {ca.sum():,} / 시험 {te.sum():,})")
        print(f"    시험 RMSE = {rmse_te:.2f} 단위/일")
        for key, ko in (("norm", "정규화"), ("plain", "비정규화(C2)")):
            r = res[key]
            rho_txt = "  정의불가(상수폭)" if np.isnan(r["rho"]) else f"{r['rho']:+.3f}"
            print(f"    {ko:12s} 포함률 {r['cover']*100:5.1f}% · 평균 폭 {r['width']:6.2f} · "
                  f"폭 변동계수 {r['width_cv']:.3f} · 반폭-실오차 순위상관 {rho_txt}")
            print(f"    {'':12s} 수요 3분위별 포함률 "
                  + " / ".join(f"{v*100:.1f}%" for v in r["by_tier"])
                  + f" (편차 {r['tier_spread']*100:.1f}%p)")
        cs = res["norm"]["by_store"]
        print("    점포별 조건부 포함률(정규화): "
              + ", ".join(f"{k} {v*100:.1f}%" for k, v in cs.items()))
        dr = res["drift"]
        print(f"    표류 점검: 평균 잔차 보정 {dr['bias_calib']:+.2f} → 시험 "
              f"{dr['bias_test']:+.2f} (평균 수요 {dr['mean_calib']:.1f} → {dr['mean_test']:.1f})")
    return res


# ===============================================================
# 4. 발주 정책의 실현 손익
# ===============================================================
def newsvendor_loss(order, actual, Cu, Co):
    short = np.maximum(actual - order, 0)
    over = np.maximum(order - actual, 0)
    return Cu * short.sum(), Co * over.sum()


def score_policies(res, Cu, Co, cr):
    """세 정책을 시험 집합에서 채점한다. 발주량은 정수로 반올림."""
    act = res["act_test"]
    q_of = res["norm"]["q_of"]
    policies = {
        "P1 점추정": np.rint(res["pred_test"]),
        "P2 구간 상한(90%)": np.rint(res["norm"]["hi"]),
        "P3 newsvendor(CR)": np.rint(q_of(cr)),
    }
    rows = []
    for name, order in policies.items():
        order = np.maximum(order, 0)
        cu_loss, co_loss = newsvendor_loss(order, act, Cu, Co)
        fill = float(np.mean(act <= order))          # 실현 서비스 수준
        rows.append(dict(정책=name, 결품손실=cu_loss, 과잉손실=co_loss,
                         총손실=cu_loss + co_loss, 실현서비스수준=fill,
                         평균발주=float(order.mean())))
    return pd.DataFrame(rows), policies


# ===============================================================
# 5. 대조 예측기 — 계절 나이브(같은 요일 전주 값)
# ===============================================================
def run_seasonal_naive(df, item):
    """LSTM 대신 '지난주 같은 요일'로 예측하고, 같은 conformal 절차를 붙인다."""
    d = df[df["item"] == item].copy()
    n_days = d["day"].nunique()
    n_train_day = int(n_days * TRAIN_FRAC)
    n_calib_day = int(n_days * (TRAIN_FRAC + CALIB_FRAC))
    stores = sorted(d["store_id"].unique())

    pred, act, si, day = [], [], [], []
    for s, sid in enumerate(stores):
        v = d[d["store_id"] == sid].sort_values("day")["demand"].to_numpy(dtype=float)
        for t in range(WINDOW, len(v)):
            pred.append(v[t - 7])
            act.append(v[t])
            si.append(s)
            day.append(t)
    pred, act = np.array(pred), np.array(act)
    si, day = np.array(si), np.array(day)
    tr, ca, te = day < n_train_day, (day >= n_train_day) & (day < n_calib_day), day >= n_calib_day

    sig_tr, fitted = fit_scale(pred[tr], np.abs(act[tr] - pred[tr]), si[tr], len(stores))
    apply = fitted[2]
    e_ca = (act[ca] - pred[ca]) / apply(pred[ca], si[ca])
    sig_te = apply(pred[te], si[te])

    def q_of(tau):
        return pred[te] + conformal_quantile(e_ca, tau) * sig_te

    return dict(rmse=float(np.sqrt(np.mean((pred[te] - act[te]) ** 2))),
                q_of=q_of, act=act[te], pred=pred[te])


# ===============================================================
# 6. 그림
# ===============================================================
def draw_figure(res_by_item, cr_by_item, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, item in zip(axes, res_by_item):
        res = res_by_item[item]
        m = res["meta_test"]
        s0 = m["store_id"].iloc[0]
        sel = (m["store_id"] == s0).to_numpy()
        n_show = 90
        idx = np.where(sel)[0][:n_show]
        x = np.arange(len(idx))
        ax.plot(x, res["act_test"][idx], color="0.35", lw=1.2, label="실현 수요")
        ax.plot(x, np.rint(res["pred_test"][idx]), color="tab:blue", lw=1.0,
                ls="--", label="P1 점추정 발주")
        ax.plot(x, np.rint(res["norm"]["hi"][idx]), color="tab:red", lw=1.0,
                ls=":", label="P2 구간 상한 발주")
        ax.plot(x, np.rint(res["norm"]["q_of"](cr_by_item[item])[idx]), color="tab:green",
                lw=1.4, label=f"P3 newsvendor 발주(CR={cr_by_item[item]:.3f})")
        ax.set_title(f"{item} — 점포 {s0}, 시험 구간 {n_show}일")
        ax.set_xlabel("시험 구간 경과일")
        ax.set_ylabel("수량(단위)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("같은 예측, 정반대 발주 — 임계비가 구간의 어느 지점을 고르는가", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ===============================================================
def main():
    np.random.seed(SEED)
    print("=" * 72)
    print("매출 예측과 발주량 결정 (7-3, LSTM + 분할 conformal → newsvendor 임계비)")
    print("=" * 72)
    print(f"장치: {DEVICE}")

    het = load_panel("store_demand.parquet")
    hom = load_panel("store_demand_homoskedastic.parquet")
    items = list(COST.keys())

    print("\n[비용 가정값] 전부 가정이며, 결론은 금액이 아니라 임계비의 방향에 둔다")
    cr_by_item = {}
    rows = []
    for it in items:
        Cu, Co = COST[it]["Cu"], COST[it]["Co"]
        cr = Cu / (Cu + Co)
        cr_by_item[it] = cr
        rows.append(dict(품목=it, Cu=Cu, Co=Co, 임계비=round(cr, 4),
                         설명=COST[it]["설명"]))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 본 데이터 ----------
    print("\n" + "-" * 72)
    print("[1] 본 데이터(이분산) — 3분할 split conformal")
    print("-" * 72)
    res_het = {it: run_item(het, it, "이분산") for it in items}

    # ---------- 대조군 C1: 등분산 ----------
    print("\n" + "-" * 72)
    print("[2] 대조군 C1 — 등분산 데이터(잡음 크기 구조만 제거)")
    print("-" * 72)
    res_hom = {it: run_item(hom, it, "등분산C1") for it in items}

    print("\n[대조군 C1·C2 판정] 정규화의 이득이 정말 이분산에서 오는가")
    c1_rows = []
    for it in items:
        for label, bank in (("이분산(본)", res_het), ("등분산(C1)", res_hom)):
            r_n, r_p = bank[it]["norm"], bank[it]["plain"]
            c1_rows.append(dict(
                품목=it, 데이터=label,
                폭변동계수=round(r_n["width_cv"], 3),
                반폭_실오차_순위상관=round(r_n["rho"], 3),
                분위편차_정규화=round(r_n["tier_spread"] * 100, 1),
                분위편차_비정규화C2=round(r_p["tier_spread"] * 100, 1)))
    print(pd.DataFrame(c1_rows).to_string(index=False))
    print("  폭변동계수 = 구간 폭이 관측마다 얼마나 달라지는가(등분산이면 0에 가까워야 한다)")
    print("  분위편차 = 수요 3분위별 조건부 포함률의 최대−최소(%p). 작을수록 고른 구간")

    # ---------- 발주 정책 채점 ----------
    print("\n" + "-" * 72)
    print("[3] 세 발주 정책의 실현 손익 (시험 집합, 본 데이터)")
    print("-" * 72)
    policy_rows = []
    for it in items:
        Cu, Co = COST[it]["Cu"], COST[it]["Co"]
        tab, _ = score_policies(res_het[it], Cu, Co, cr_by_item[it])
        base = tab.loc[tab["정책"] == "P3 newsvendor(CR)", "총손실"].iloc[0]
        tab.insert(0, "품목", it)
        tab["P3대비"] = (tab["총손실"] - base)
        print(f"\n  {it} (CR = {cr_by_item[it]:.3f}, 시험 {res_het[it]['n_test']:,}건)")
        show = tab.copy()
        for c in ["결품손실", "과잉손실", "총손실", "P3대비"]:
            show[c] = show[c].map(lambda v: f"{v:,.0f}")
        show["실현서비스수준"] = show["실현서비스수준"].map(lambda v: f"{v*100:.1f}%")
        show["평균발주"] = show["평균발주"].round(1)
        print(show.drop(columns=["품목"]).to_string(index=False))
        policy_rows.append(tab)
    policy_df = pd.concat(policy_rows, ignore_index=True)

    # ---------- 임계비 민감도 ----------
    print("\n" + "-" * 72)
    print("[4] 임계비 민감도 — 같은 예측에서 CR만 흔들면 발주와 손실이 얼마나 움직이나")
    print("-" * 72)
    sens_rows = []
    for it in items:
        Cu, Co = COST[it]["Cu"], COST[it]["Co"]
        res = res_het[it]
        true_cr = cr_by_item[it]
        for cr in [0.10, 0.25, 0.375, 0.50, 0.75, 0.909, 0.95]:
            order = np.maximum(np.rint(res["norm"]["q_of"](cr)), 0)
            cu_l, co_l = newsvendor_loss(order, res["act_test"], Cu, Co)
            sens_rows.append(dict(품목=it, 가정CR=cr, 평균발주=round(float(order.mean()), 1),
                                  총손실=cu_l + co_l))
        opt = min(r["총손실"] for r in sens_rows if r["품목"] == it)
        for r in sens_rows:
            if r["품목"] == it:
                r["최적대비초과"] = r["총손실"] - opt
                r["참CR"] = round(true_cr, 3)
    sens = pd.DataFrame(sens_rows)
    for it in items:
        sub = sens[sens["품목"] == it].copy()
        print(f"\n  {it} (참 CR = {cr_by_item[it]:.3f})")
        sub["총손실"] = sub["총손실"].map(lambda v: f"{v:,.0f}")
        sub["최적대비초과"] = sub["최적대비초과"].map(lambda v: f"{v:+,.0f}")
        print(sub[["가정CR", "평균발주", "총손실", "최적대비초과"]].to_string(index=False))

    # ---------- 비용비 오지정 대 예측 개선 ----------
    print("\n" + "-" * 72)
    print("[5] 비용비를 틀리는 대가 대 예측을 개선하는 이득")
    print("-" * 72)
    cmp_rows = []
    for it in items:
        Cu, Co = COST[it]["Cu"], COST[it]["Co"]
        cr = cr_by_item[it]
        res = res_het[it]

        def loss_of(order, act):
            a, b = newsvendor_loss(np.maximum(np.rint(order), 0), act, Cu, Co)
            return a + b

        best = loss_of(res["norm"]["q_of"](cr), res["act_test"])
        naive = run_seasonal_naive(het, it)
        naive_loss = loss_of(naive["q_of"](cr), naive["act"])
        sym_loss = loss_of(res["norm"]["q_of"](0.5), res["act_test"])
        other_cr = cr_by_item["우산" if it == "도시락" else "도시락"]
        swap_loss = loss_of(res["norm"]["q_of"](other_cr), res["act_test"])
        cmp_rows.append(dict(
            품목=it,
            기준_LSTM_참CR=best,
            예측열화_계절나이브=naive_loss - best,
            비용비오지정_대칭가정=sym_loss - best,
            비용비오지정_뒤바뀜=swap_loss - best,
            RMSE_LSTM=round(res["rmse"], 2), RMSE_나이브=round(naive["rmse"], 2)))
    cmp = pd.DataFrame(cmp_rows)
    show = cmp.copy()
    for c in ["기준_LSTM_참CR", "예측열화_계절나이브", "비용비오지정_대칭가정", "비용비오지정_뒤바뀜"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))
    print("  (금액은 시험 구간 전체 합계. 값이 클수록 그 잘못의 대가가 크다)")

    # 등가 오차 — 예측을 나이브로 떨어뜨린 대가와 같아지는 CR 오차 폭
    print("\n  [등가 오차] 계절 나이브로 떨어뜨린 대가와 같은 손실을 내는 CR 오차 폭")
    eq_rows = []
    for it in items:
        Cu, Co = COST[it]["Cu"], COST[it]["Co"]
        cr = cr_by_item[it]
        res = res_het[it]
        base = float(cmp.loc[cmp["품목"] == it, "기준_LSTM_참CR"].iloc[0])
        target = float(cmp.loc[cmp["품목"] == it, "예측열화_계절나이브"].iloc[0])
        grid = np.arange(0.01, 0.9951, 0.005)
        eq = {}
        for side, sub in (("아래", grid[grid < cr][::-1]), ("위", grid[grid > cr])):
            hit, worst = None, 0.0
            for c in sub:
                a, b = newsvendor_loss(np.maximum(np.rint(res["norm"]["q_of"](c)), 0),
                                       res["act_test"], Cu, Co)
                worst = max(worst, a + b - base)
                if a + b - base >= target:
                    hit = f"{abs(c - cr):.3f}"
                    break
            # 끝까지 가도 도달하지 않으면 그 방향은 비용비 오지정에 둔감하다는 뜻이다
            eq[side] = hit if hit else f"미도달(최대 초과 {worst:,.0f}원)"
        eq_rows.append(dict(품목=it, 참CR=round(cr, 3), 나이브대가=f"{target:,.0f}",
                            등가CR오차_아래=eq["아래"], 등가CR오차_위=eq["위"]))
    eq_df = pd.DataFrame(eq_rows)
    print(eq_df.to_string(index=False))
    print("  (임계비를 이 폭만큼 틀리면 예측 모델을 계절 나이브로 바꾼 것과 같은 손실이 된다)")

    # ---------- 저장 ----------
    cov_rows = []
    for it in items:
        for label, bank in (("이분산", res_het), ("등분산C1", res_hom)):
            for key, ko in (("norm", "정규화"), ("plain", "비정규화")):
                r = bank[it][key]
                cov_rows.append(dict(데이터=label, 품목=it, conformal=ko,
                                     목표포함률=TARGET_COVER, 실측포함률=round(r["cover"], 4),
                                     평균폭=round(r["width"], 3),
                                     폭변동계수=round(r["width_cv"], 4),
                                     반폭_실오차_순위상관=round(r["rho"], 4),
                                     분위별포함률_편차pp=round(r["tier_spread"] * 100, 2),
                                     점포별포함률_최소=round(min(r["by_store"].values()), 4),
                                     점포별포함률_최대=round(max(r["by_store"].values()), 4),
                                     잔차평균_보정=round(bank[it]["drift"]["bias_calib"], 3),
                                     잔차평균_시험=round(bank[it]["drift"]["bias_test"], 3),
                                     RMSE=round(bank[it]["rmse"], 3)))
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(RESULTS_DIR / "ch7_conformal_coverage.csv", index=False, encoding="utf-8-sig")
    policy_df.to_csv(RESULTS_DIR / "ch7_newsvendor_policy.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(RESULTS_DIR / "ch7_newsvendor_sensitivity.csv", index=False, encoding="utf-8-sig")

    png = RESULTS_DIR / "7-3-demand-newsvendor.png"
    draw_figure(res_het, cr_by_item, png)

    print("\n[저장] ch7_conformal_coverage.csv / ch7_newsvendor_policy.csv / "
          "ch7_newsvendor_sensitivity.csv / 7-3-demand-newsvendor.png")
    print("\n[완료] 예측구간을 임계비로 발주 결정에 옮기는 계산을 마쳤다.")


if __name__ == "__main__":
    main()
