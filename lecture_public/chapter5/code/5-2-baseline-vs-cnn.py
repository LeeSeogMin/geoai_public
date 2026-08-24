"""
5-2. 저비용 기준선 대비 CNN — 이 과업에 딥러닝이 필요한가
========================================================
5-1은 CNN끼리(SimpleCNN 대 MiniResNet), 밴드끼리(RGB·4밴드·12밴드)만 비교했다.
빠진 질문이 하나 있다. **애초에 CNN이 필요한 과업인가.**

5-1의 밴드 실험이 이미 단서를 준다. RGB 0.973 → RGB+NIR 0.984 → 12밴드 0.984.
근적외선 한 밴드가 나머지 여덟 개보다 값이 컸다. 어떤 밴드를 넣을지는 통계
문제이기 전에 물리 문제라는 뜻이고, 그렇다면 그 물리를 그대로 쓴 지수
(NDVI·NDWI·NDBI)만으로 어디까지 갈 수 있는지부터 재 봐야 한다.

비교 대상 다섯
--------------
  R1 고정 임계 규칙   : 학습 없음. NDWI>0 수역, NDVI>0.5 산림, NDBI>0 시가지, 나머지 경작지
  R2 학습된 임계      : 같은 지수 3개만 쓰되 임계를 결정트리(깊이 3)로 학습
  L  로지스틱 회귀     : 패치 요약 피처(밴드·지수의 평균과 표준편차 30개)
  RF 랜덤 포레스트     : 같은 요약 피처
  CNN SimpleCNN       : 12밴드 32×32 패치 원본 (5-1과 같은 구조)

무엇을 맞추는가
--------------
`5-1`과 **같은 패치, 같은 클래스 균형, 같은 공간 타일 분할(GroupKFold)**로
채점한다. 그래야 이 표의 숫자를 5-1의 숫자와 나란히 읽을 수 있다.
`5-1` 파일은 건드리지 않는다(그 로그 수치가 본문 여러 곳에 인용돼 있다).

무엇을 함께 재는가
-----------------
정확도만으로는 도입 판단을 못 한다. 학습·추론 시간, 모델 크기, 그리고
분석 2(`5-3`)가 비용 계산에 쓸 성능 격차를 CSV로 넘긴다.

주의: 성능 차이의 부호를 해석하기 전에 실행 간 변동부터 잰다. CNN은 시드를
바꿔 세 번 돌려 그 폭을 함께 보고한다(5-1의 재현성 점검과 같은 취지).

실행 (프로젝트 루트, 통합 .venv):
    python lecture_practice/chapter5/code/5-2-baseline-vs-cnn.py
    # 데이터가 없으면 먼저: python lecture_practice/chapter4/code/4-0-data-download.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from _s2_data import BAND_NAMES, extract_patches

SEED = 42
EPOCHS, BATCH = 15, 32
CNN_SEEDS = [42, 0, 1]          # 실행 간 변동을 재려고 시드를 바꿔 반복
RESULT_DIR = Path(__file__).resolve().parent.parent / "results"

np.random.seed(SEED)
torch.manual_seed(SEED)

IDX = {b: i for i, b in enumerate(BAND_NAMES)}

print("=" * 62)
print("저비용 기준선 대비 CNN: 이 과업에 딥러닝이 필요한가")
print("=" * 62)


# ============================================================
# 1. 5-1과 동일한 데이터셋 구성
# ============================================================
# 아래 블록은 5-1의 패치 선별·균형 표집을 그대로 재현한다. 같은 시드, 같은
# 호출 순서를 지켜야 두 스크립트가 같은 724장을 본다.
print("\n--- 1. 데이터 (5-1과 동일한 구성) ---")
patches, labels, tiles, info = extract_patches(patch=32, stride=24, purity=0.6)
all_names = info["class_names"]
MIN_PATCHES = 80
keep_cls = [i for i in range(len(all_names)) if (labels == i).sum() >= MIN_PATCHES]
rng = np.random.default_rng(SEED)
cap = min((labels == i).sum() for i in keep_cls)
sel_idx, new_label = [], {}
for new_i, c in enumerate(keep_cls):
    new_label[c] = new_i
    idx = np.where(labels == c)[0]
    sel_idx.append(rng.choice(idx, size=cap, replace=False))
sel = np.concatenate(sel_idx)
X_patch = patches[sel]
y = np.array([new_label[labels[i]] for i in sel])
groups = tiles[sel]
class_names = [all_names[c] for c in keep_cls]
n_classes = len(class_names)
print(f"  패치 {len(y)}장, {n_classes}클래스 {class_names}, 클래스당 {cap}개")
print(f"  입력 텐서 {X_patch.shape[1:]}, 공간 타일 {len(np.unique(groups))}개")


# ============================================================
# 2. 지수와 요약 피처
# ============================================================
def patch_indices(cube):
    """패치별 NDVI·NDWI·NDBI를 픽셀 단위로 계산한다. (N,3,P,P)"""
    eps = 1e-6
    red, nir = cube[:, IDX["B04"]], cube[:, IDX["B08"]]
    green, swir = cube[:, IDX["B03"]], cube[:, IDX["B11"]]
    ndvi = (nir - red) / (nir + red + eps)      # 식생: 엽록소가 NIR을 강하게 반사
    ndwi = (green - nir) / (green + nir + eps)  # 수역: 물은 NIR을 거의 흡수
    ndbi = (swir - nir) / (swir + nir + eps)    # 시가지: 건조·불투수면에서 SWIR이 높음
    return np.stack([ndvi, ndwi, ndbi], axis=1)


IDX3 = patch_indices(X_patch)
idx_mean = IDX3.mean(axis=(2, 3))                    # (N,3) 지수 평균 — 규칙이 쓰는 값
# 요약 피처: 12밴드 평균·표준편차 + 지수 3종 평균·표준편차 = 30개
# 표준편차를 넣는 이유는 질감의 값싼 대리 지표이기 때문이다. CNN이 이기더라도
# "질감을 봐서 이겼다"고 말하려면 질감의 요약본을 이미 준 기준선을 이겨야 한다.
X_sum = np.concatenate([
    X_patch.mean(axis=(2, 3)), X_patch.std(axis=(2, 3)),
    idx_mean, IDX3.std(axis=(2, 3)),
], axis=1)
feat_names = ([f"{b}_mean" for b in BAND_NAMES] + [f"{b}_sd" for b in BAND_NAMES]
              + ["NDVI_mean", "NDWI_mean", "NDBI_mean"]
              + ["NDVI_sd", "NDWI_sd", "NDBI_sd"])
print(f"  요약 피처 {X_sum.shape[1]}개 (밴드 평균·표준편차 + 지수 평균·표준편차)")

NAME2I = {n: i for i, n in enumerate(class_names)}
print("  지수 평균 (클래스별) — 규칙이 기대는 물리가 실제로 갈라지는지 확인")
print(f"    {'클래스':6s} | {'NDVI':>7s} | {'NDWI':>7s} | {'NDBI':>7s}")
for i, n in enumerate(class_names):
    m = idx_mean[y == i].mean(axis=0)
    print(f"    {n:6s} | {m[0]:+7.3f} | {m[1]:+7.3f} | {m[2]:+7.3f}")


# ============================================================
# 3. R1 — 고정 임계 규칙 (학습 없음)
# ============================================================
def rule_predict(idx_mean_arr):
    """분광지수의 관행적 임계만으로 네 유형을 가른다. 학습 파라미터가 0개다.

    순서가 중요하다. 물은 NIR을 거의 흡수해 가장 확실히 갈리므로 먼저 떼고,
    그다음 식생을 떼고, 남은 것 중 건조·불투수면을 시가지로 본다.
    """
    ndvi, ndwi, ndbi = idx_mean_arr[:, 0], idx_mean_arr[:, 1], idx_mean_arr[:, 2]
    out = np.full(len(idx_mean_arr), NAME2I["경작지"])      # 기본값
    out[(ndvi <= 0.5) & (ndbi > 0.0)] = NAME2I["시가지"]
    out[ndvi > 0.5] = NAME2I["산림"]
    out[ndwi > 0.0] = NAME2I["수역"]
    return out


# ============================================================
# 4. CNN — 5-1과 같은 구조
# ============================================================
class SimpleCNN(nn.Module):
    """5-1의 SimpleCNN과 동일한 구조(96,804 파라미터). 비교를 위해 그대로 옮긴다."""

    def __init__(self, in_channels=12, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    """크로스 플랫폼 디바이스 자동 감지 (CUDA → MPS → CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
XT = torch.from_numpy(np.clip(X_patch, 0, 1)).float()
YT = torch.from_numpy(y).long()


def cnn_fold(tr, va, seed):
    """한 폴드를 학습하고 (예측, 학습초, 추론초)를 돌려준다."""
    torch.manual_seed(seed)
    model = SimpleCNN(X_patch.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(XT[tr], YT[tr]), batch_size=BATCH, shuffle=True)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
    fit_s = time.perf_counter() - t0
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(XT[va].to(DEVICE)).argmax(1).cpu().numpy()
    return pred, fit_s, time.perf_counter() - t0


# ============================================================
# 5. 같은 공간 타일 분할로 다섯 방법을 채점
# ============================================================
print("\n--- 2. 공간 타일 분할(GroupKFold 5겹)로 다섯 방법 채점 ---")
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.zeros(len(y)), groups=groups))

results = {}
oof_pred = {}


def score_sklearn(name, make_model, X, needs_scale=False):
    """폴드별로 학습·예측하고 지표와 시간을 모은다."""
    accs, f1s, fit_t, inf_t = [], [], 0.0, 0.0
    oof = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        Xtr, Xva = X[tr], X[va]
        if needs_scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        m = make_model()
        t0 = time.perf_counter()
        m.fit(Xtr, y[tr])
        fit_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        p = m.predict(Xva)
        inf_t += time.perf_counter() - t0
        oof[va] = p
        accs.append((p == y[va]).mean())
        f1s.append(f1_score(y[va], p, average="macro", zero_division=0))
    results[name] = {"acc": np.mean(accs), "acc_sd": np.std(accs),
                     "f1": np.mean(f1s), "f1_sd": np.std(f1s),
                     "f1_folds": np.array(f1s),
                     "fit_s": fit_t, "inf_s": inf_t}
    oof_pred[name] = oof
    print(f"  {name:22s} Acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
          f"Macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
          f"학습={fit_t:.2f}초 추론={inf_t:.3f}초")


# R1 고정 임계 규칙: 학습이 없으므로 폴드마다 같은 규칙을 검증 부분에만 적용
r1_all = rule_predict(idx_mean)
accs, f1s = [], []
t0 = time.perf_counter()
for _ in range(5):
    rule_predict(idx_mean)
r1_inf = (time.perf_counter() - t0) / 5
for tr, va in folds:
    accs.append((r1_all[va] == y[va]).mean())
    f1s.append(f1_score(y[va], r1_all[va], average="macro", zero_division=0))
results["R1 고정 임계 규칙"] = {"acc": np.mean(accs), "acc_sd": np.std(accs),
                          "f1": np.mean(f1s), "f1_sd": np.std(f1s),
                          "f1_folds": np.array(f1s), "fit_s": 0.0, "inf_s": r1_inf}
oof_pred["R1 고정 임계 규칙"] = r1_all
print(f"  {'R1 고정 임계 규칙':22s} Acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
      f"Macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
      f"학습=0.00초 추론={r1_inf:.3f}초")

score_sklearn("R2 학습된 임계(트리)",
              lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED), idx_mean)
score_sklearn("L  로지스틱 회귀",
              lambda: LogisticRegression(max_iter=2000, random_state=SEED),
              X_sum, needs_scale=True)
score_sklearn("RF 랜덤 포레스트",
              lambda: RandomForestClassifier(n_estimators=300, random_state=SEED,
                                             n_jobs=-1), X_sum)

# CNN — 시드 3개로 반복해 실행 간 변동을 함께 잰다
cnn_runs = []
for si, seed in enumerate(CNN_SEEDS):
    accs, f1s, fit_t, inf_t = [], [], 0.0, 0.0
    oof = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        p, ft, it = cnn_fold(tr, va, seed)
        fit_t += ft
        inf_t += it
        oof[va] = p
        accs.append((p == y[va]).mean())
        f1s.append(f1_score(y[va], p, average="macro", zero_division=0))
    cnn_runs.append({"seed": seed, "acc": np.mean(accs), "f1": np.mean(f1s),
                     "f1_folds": np.array(f1s), "fit_s": fit_t, "inf_s": inf_t,
                     "oof": oof, "acc_sd": np.std(accs), "f1_sd": np.std(f1s)})
    print(f"  {'CNN SimpleCNN':16s} seed={seed:<3d} Acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
          f"Macro-F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
          f"학습={fit_t:.2f}초 추론={inf_t:.3f}초")

base_run = cnn_runs[0]                      # 시드 42를 대표 실행으로 삼는다
results["CNN SimpleCNN"] = {k: base_run[k] for k in
                            ("acc", "acc_sd", "f1", "f1_sd", "f1_folds", "fit_s", "inf_s")}
oof_pred["CNN SimpleCNN"] = base_run["oof"]
cnn_f1_range = (min(r["f1"] for r in cnn_runs), max(r["f1"] for r in cnn_runs))
print(f"  CNN 시드 3개의 Macro-F1 범위: {cnn_f1_range[0]:.3f} ~ {cnn_f1_range[1]:.3f} "
      f"(폭 {cnn_f1_range[1] - cnn_f1_range[0]:.3f})")

# ============================================================
# 6. 비용·복잡도까지 나란히
# ============================================================
COMPLEXITY = {
    "R1 고정 임계 규칙": ("0", "없음", "지수 3개 계산 + if 4줄"),
    "R2 학습된 임계(트리)": ("트리 노드 ≤15", "라벨 필요", "지수 3개 + 결정트리"),
    "L  로지스틱 회귀": ("124", "라벨 필요", "요약 피처 30개 + 표준화"),
    "RF 랜덤 포레스트": ("트리 300개", "라벨 필요", "요약 피처 30개"),
    "CNN SimpleCNN": ("96,804", "라벨 필요", "GPU 권장 + 학습 파이프라인"),
}
print("\n--- 3. 성능·비용·복잡도 ---")
print(f"  {'방법':22s} | {'Acc':>5s} | {'Macro-F1':>8s} | {'학습초':>7s} | "
      f"{'추론초':>7s} | {'파라미터':>12s} | 구현")
order = ["R1 고정 임계 규칙", "R2 학습된 임계(트리)", "L  로지스틱 회귀",
         "RF 랜덤 포레스트", "CNN SimpleCNN"]
for k in order:
    r, c = results[k], COMPLEXITY[k]
    print(f"  {k:22s} | {r['acc']:5.3f} | {r['f1']:8.3f} | {r['fit_s']:7.2f} | "
          f"{r['inf_s']:7.3f} | {c[0]:>12s} | {c[2]}")

# ============================================================
# 7. CNN이 무엇을 더 맞히는가 — 클래스별로
# ============================================================
print("\n--- 4. 클래스별 F1 (out-of-fold) ---")
print(f"  {'방법':22s} | " + " | ".join(f"{n:>6s}" for n in class_names))
for k in order:
    pf = precision_recall_fscore_support(y, oof_pred[k], labels=range(n_classes),
                                         zero_division=0)[2]
    print(f"  {k:22s} | " + " | ".join(f"{v:6.3f}" for v in pf))

best_base = max(["R1 고정 임계 규칙", "R2 학습된 임계(트리)",
                 "L  로지스틱 회귀", "RF 랜덤 포레스트"], key=lambda k: results[k]["f1"])
gap_f1 = results["CNN SimpleCNN"]["f1"] - results[best_base]["f1"]
gap_acc = results["CNN SimpleCNN"]["acc"] - results[best_base]["acc"]
paired = results["CNN SimpleCNN"]["f1_folds"] - results[best_base]["f1_folds"]

print("\n--- 5. CNN 대 최고 기준선 ---")
print(f"  최고 기준선: {best_base} (Macro-F1 {results[best_base]['f1']:.3f})")
print(f"  CNN Macro-F1 {results['CNN SimpleCNN']['f1']:.3f} → 격차 {gap_f1:+.3f} "
      f"(Accuracy {gap_acc:+.3f})")
print(f"  폴드별 격차: " + ", ".join(f"{d:+.3f}" for d in paired))
print(f"  폴드별 격차 범위 {paired.min():+.3f} ~ {paired.max():+.3f}, "
      f"CNN 시드 변동 폭 {cnn_f1_range[1] - cnn_f1_range[0]:.3f}")
if abs(gap_f1) <= (cnn_f1_range[1] - cnn_f1_range[0]):
    print("  → 격차가 CNN 자신의 실행 간 변동 폭 이하다. 부호를 해석하지 않는다.")
else:
    print("  → 격차가 실행 간 변동 폭보다 크다. 방향을 말할 수 있다.")
print(f"  학습 시간 비: CNN {results['CNN SimpleCNN']['fit_s']:.2f}초 대 "
      f"{best_base} {results[best_base]['fit_s']:.2f}초")

# ============================================================
# 8. 분석 2(5-3)로 넘길 실측 격차
# ============================================================
RESULT_DIR.mkdir(exist_ok=True)
rows = []
for k in order:
    r, c = results[k], COMPLEXITY[k]
    rows.append({"방법": k, "정확도": r["acc"], "정확도_sd": r["acc_sd"],
                 "Macro_F1": r["f1"], "Macro_F1_sd": r["f1_sd"],
                 "학습_초": r["fit_s"], "추론_초": r["inf_s"],
                 "파라미터": c[0], "라벨_필요": c[1], "구현": c[2]})
df = pd.DataFrame(rows)
df.to_csv(RESULT_DIR / "ch5_baseline_comparison.csv", index=False, encoding="utf-8-sig")

handoff = pd.DataFrame([{
    "최고_기준선": best_base,
    "기준선_정확도": results[best_base]["acc"],
    "기준선_MacroF1": results[best_base]["f1"],
    "CNN_정확도": results["CNN SimpleCNN"]["acc"],
    "CNN_MacroF1": results["CNN SimpleCNN"]["f1"],
    "정확도_격차": gap_acc,
    "MacroF1_격차": gap_f1,
    "CNN_시드변동_폭": cnn_f1_range[1] - cnn_f1_range[0],
    "규칙_정확도": results["R1 고정 임계 규칙"]["acc"],
    "규칙_MacroF1": results["R1 고정 임계 규칙"]["f1"],
    "CNN_학습_초": results["CNN SimpleCNN"]["fit_s"],
    "패치_수": len(y),
}])
handoff.to_csv(RESULT_DIR / "ch5_perf_handoff.csv", index=False, encoding="utf-8-sig")
print(f"\n  저장: results/ch5_baseline_comparison.csv, results/ch5_perf_handoff.csv")
print("  (5-3이 ch5_perf_handoff.csv의 실측 격차를 읽어 도입 손익분기를 계산한다)")

print("\n--- 6. 이 비교가 말하는 것과 말하지 못하는 것 ---")
print("  말하는 것: 이 과업(순수 패치·네 유형·단일 장면)에서 분광지수 기준선이")
print("    어디까지 가고 CNN이 그 위에 무엇을 얹는지. 그리고 그 차이가 CNN 자신의")
print("    실행 간 변동보다 큰지 작은지.")
print("  말하지 못하는 것: 다른 과업에서의 우열. 유형이 세분되거나 분광이 겹치거나")
print("    경계·혼합 패치가 포함되면 지수 규칙이 먼저 무너진다. 이 결과는")
print("    '패치를 순수한 것만 골랐을 때'라는 조건 위에 서 있다.")

print("\n[완료] 기준선 대비 CNN 비교를 마쳤다.")
