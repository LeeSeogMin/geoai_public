"""
7장 실습 2: 시공간 예측과 불확실성 정량화 (LSTM + MC Dropout)
=================================================================
정책 질문: "다음 시점 교통량(수요)은 얼마이고, 그 예측을 얼마나 믿을 수 있는가?"

시공간 예측의 정책 가치는 '앞으로 어떻게 될 것인가'에 답하는 것이다.
그러나 점추정만으로는 위험하다 — 신뢰구간이 자원 배치 결정을 좌우한다.
불확실성이 큰 시점·지역은 추가 관측이나 보수적 대응이 필요하다.

모델: 소형 LSTM(과거 24시간 → 다음 1시간 예측), CPU 최적화(적은 데이터·소수 에폭).
불확실성: MC Dropout(추론 시 dropout 유지 → 다수 forward로 예측 분포 근사,
          Gal & Ghahramani 2016). 점추정 vs 90% 신뢰구간 비교.

평가: RMSE(점추정 정확도), 구간 포함률(coverage, 90% 목표), 평균 구간 폭.

데이터: 미리 준비한 교육용 합성 교통 시계열(일·주 주기 + 이분산 잡음)을
  불러와 쓴다. 분석·예측은 실제 계산값.
실행:
    python 7-0-simdata-prep.py            # 최초 1회: 데이터 준비
    python 7-2-spatiotemporal-uncertainty.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)               # 모델 초기화·MC Dropout 재현성(데이터는 저장본 사용)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")        # 소형 모델 — CPU로 재현성 확보
WINDOW = 24                          # 과거 24시간
HOURS = 1200                         # 총 1200시간(약 50일)
EPOCHS = 15
MC_SAMPLES = 50                      # MC Dropout 표본 수


def load_traffic():
    """미리 준비한 합성 교통량 시계열을 불러온다.

    일 주기(24h) + 주 주기(168h)에 낮 피크 시간대일수록 큰 이분산 잡음이
    더해진 구조다. 시간대에 따라 잡음 크기가 달라져 불확실성 구조를 담는다.
    """
    path = DATA_DIR / "traffic_volume.npy"
    if not path.exists():
        raise SystemExit(
            f"데이터가 없습니다: {path}\n먼저 실행: python 7-0-simdata-prep.py")
    return np.load(path)


def make_windows(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = np.array(X, dtype=np.float32)[:, :, None]    # (N, window, 1)
    y = np.array(y, dtype=np.float32)[:, None]
    return X, y


class LSTMRegressor(nn.Module):
    def __init__(self, hidden=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)              # MC Dropout: 추론 시에도 활성화
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.fc(h)


def main():
    print("=" * 64)
    print("시공간 예측과 불확실성 정량화 (7-2, LSTM + MC Dropout)")
    print("=" * 64)
    series = load_traffic()
    # 정규화(학습 안정)
    mu, sd = series.mean(), series.std()
    norm = (series - mu) / sd
    X, y = make_windows(norm, WINDOW)
    n_train = int(len(X) * 0.7)                      # 시계열 분할(미래 누수 차단)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    print(f"\n데이터: {HOURS}시간 합성 교통량, 윈도우 {WINDOW}h → 다음 1h 예측")
    print(f"학습 {len(Xtr)} / 검증 {len(Xte)} (앞 70% 학습, 시계열 누수 차단)")

    model = LSTMRegressor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)

    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = lossf(pred, ytr_t)
        loss.backward()
        opt.step()
    print(f"학습 완료(에폭 {EPOCHS}, 최종 train MSE {loss.item():.4f})")

    # ---- aleatoric(데이터 잡음): 학습 잔차의 분산으로 추정 ----
    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_t).squeeze(-1).numpy() * sd + mu
    tr_actual = ytr.squeeze(-1) * sd + mu
    aleatoric_var = float(np.var(tr_actual - tr_pred))     # 데이터 잡음 분산(역정규화 단위)

    # ---- MC Dropout 추론(epistemic): dropout 유지한 채 다수 forward ----
    Xte_t = torch.tensor(Xte)
    model.train()                                    # ★ dropout 활성 유지(MC Dropout 핵심)
    samples = np.zeros((MC_SAMPLES, len(Xte)))
    with torch.no_grad():
        for s in range(MC_SAMPLES):
            samples[s] = model(Xte_t).squeeze(-1).numpy()
    samples = samples * sd + mu                      # 역정규화
    yte_real = yte.squeeze(-1) * sd + mu
    pred_mean = samples.mean(axis=0)
    epistemic_var = samples.var(axis=0)              # 모델 불확실성

    # 예측 구간 = epistemic + aleatoric (정규 근사, 90% → ±1.645σ)
    total_std = np.sqrt(epistemic_var + aleatoric_var)
    pred_lo = pred_mean - 1.645 * total_std
    pred_hi = pred_mean + 1.645 * total_std

    rmse = float(np.sqrt(np.mean((pred_mean - yte_real) ** 2)))
    coverage = float(np.mean((yte_real >= pred_lo) & (yte_real <= pred_hi)) * 100)
    width = float(np.mean(pred_hi - pred_lo))
    ep_share = float(np.mean(epistemic_var) / (np.mean(epistemic_var) + aleatoric_var) * 100)
    print("\n[점추정] RMSE = {:.2f} 대/시간".format(rmse))
    print("[불확실성] 두 성분 결합: epistemic(모델) + aleatoric(데이터 잡음)")
    print("           평균 표준편차 epistemic {:.2f} / aleatoric {:.2f} (epistemic 비중 {:.0f}%)"
          .format(float(np.sqrt(np.mean(epistemic_var))), float(np.sqrt(aleatoric_var)), ep_share))
    print("           90% 구간 포함률(coverage) = {:.1f}% (목표 90%)".format(coverage))
    print("           평균 구간 폭 = {:.2f} 대/시간".format(width))
    print("  ※ MC Dropout만 쓰면(epistemic만) 데이터 잡음을 놓쳐 구간이 좁고 과소포함된다.")

    # 불확실성이 큰 시점(구간 폭 상위)
    order = np.argsort(-(pred_hi - pred_lo))
    print("\n[정책 연계] 예측 불확실성이 큰 상위 5개 시점(추가 관측·보수적 대응 대상)")
    head = pd.DataFrame({
        "test_idx": order[:5],
        "pred_mean": pred_mean[order[:5]].round(1),
        "interval_width": (pred_hi - pred_lo)[order[:5]].round(1),
        "actual": yte_real[order[:5]].round(1),
    })
    print(head.to_string(index=False))

    out = pd.DataFrame({
        "test_idx": np.arange(len(yte_real)),
        "actual": yte_real.round(2), "pred_mean": pred_mean.round(2),
        "pred_lo": pred_lo.round(2), "pred_hi": pred_hi.round(2),
    })
    csv_path = RESULTS_DIR / "traffic_uncertainty.csv"
    out.to_csv(csv_path, index=False)
    print(f"\n  예측·신뢰구간 저장 → {csv_path.name}")
    print("  ※ 정책 함의: 점추정이 같아도 구간이 넓은 시점은 의사결정 위험이 크다.")
    print("    좁은 구간 → 확정 대응, 넓은 구간 → 추가 관측·완충 자원 확보.")
    print("\n[완료] 시공간 예측·불확실성 정량화를 마쳤다.")


if __name__ == "__main__":
    main()
