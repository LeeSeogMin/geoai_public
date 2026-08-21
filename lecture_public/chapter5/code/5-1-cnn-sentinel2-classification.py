"""
5-1. CNN 기반 위성영상 분류 실습 (실제 13→12밴드 Sentinel-2 패치)
================================================================
실제 Sentinel-2 L2A 클립(춘천, 2021-04-07)에서 32×32 패치를 잘라
장면 분류 CNN을 학습한다. 라벨은 ESA WorldCover 2021 다수결로 붙이고,
같은 타일의 이웃 패치가 학습·검증에 갈라지지 않게 공간 분할한다.

- Sentinel-2 MSI 센서는 13개 밴드를 관측하지만, L2A 산출물에는
  지표 정보가 없는 B10(권운)이 제외되어 12개 밴드가 입력이 된다.
- 패치는 (밴드 × 행 × 열) = (12, 32, 32) 텐서로, 12개 숫자 배열을
  채널로 쌓아 CNN에 넣는다(그림 파일을 넣는 것이 아니다).

데이터 흐름:
  실제 GeoTIFF → 유효·구름 마스크 → 32×32 패치 추출
  → WorldCover 다수결 라벨 → 클래스 균형 표집
  → 공간 타일 분할(train/val) → CNN 학습 → Random vs Spatial 비교

실행 방법 (프로젝트 루트, 통합 .venv):
    source .venv/bin/activate
    python practice/chapter5/code/5-1-cnn-sentinel2-classification.py
    # 데이터가 없으면 먼저: python practice/chapter4/code/4-0-data-download.py

주의: CPU에서도 수 분 내 끝나도록 패치 수·에폭을 축소했다.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset, Subset

from _s2_data import extract_patches

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("CNN 기반 위성영상 분류: 실제 12밴드 Sentinel-2 패치")
print("=" * 60)

# ============================================================
# 1. 실제 패치 추출 + 클래스 균형
# ============================================================
print("\n--- 1. 실제 패치 추출 (WorldCover 라벨) ---")
patches, labels, tiles, info = extract_patches(patch=32, stride=24, purity=0.6)
all_names = info["class_names"]
print(f"  추출 패치: {len(patches)}개, shape={patches.shape[1:]} (밴드,행,열)")
for i, n in enumerate(all_names):
    print(f"    {n}: {(labels == i).sum()}개", end="")
print()

# 순수 패치가 너무 적은 클래스(초지)는 제외하고, 나머지를 균형 표집한다.
MIN_PATCHES = 80
keep_cls = [i for i in range(len(all_names)) if (labels == i).sum() >= MIN_PATCHES]
rng = np.random.default_rng(42)
cap = min((labels == i).sum() for i in keep_cls)  # 최소 클래스에 맞춰 균형
sel_idx, new_label = [], {}
for new_i, c in enumerate(keep_cls):
    new_label[c] = new_i
    idx = np.where(labels == c)[0]
    sel_idx.append(rng.choice(idx, size=cap, replace=False))
sel = np.concatenate(sel_idx)
X_patch = patches[sel]
y_patch = np.array([new_label[labels[i]] for i in sel])
tile_patch = tiles[sel]
class_names = [all_names[c] for c in keep_cls]
n_classes = len(class_names)
N_BANDS = X_patch.shape[1]
print(f"  분류 대상: {n_classes}클래스 {class_names}, 클래스당 {cap}개 (균형)")
print(f"  제외: {[all_names[i] for i in range(len(all_names)) if i not in keep_cls]}"
      f" (순수 패치 {MIN_PATCHES}개 미만)")
print(f"  공간 타일 수: {len(np.unique(tile_patch))}개")

# ============================================================
# 2. Dataset 정의
# ============================================================
class PatchDataset(Dataset):
    """실제 Sentinel-2 패치 데이터셋 (반사율 0~1은 이미 스케일됨)"""

    def __init__(self, patches, labels):
        self.x = torch.from_numpy(np.clip(patches, 0, 1)).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = PatchDataset(X_patch, y_patch)
print(f"\n--- 2. 데이터셋 ---")
print(f"  크기: {len(dataset)}, 입력 텐서 shape: {tuple(dataset[0][0].shape)}")

# ============================================================
# 3. 모델 정의
# ============================================================
print("\n--- 3. 모델 정의 ---")


class SimpleCNN(nn.Module):
    """12채널 입력 경량 CNN 분류기"""

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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


class MiniResNet(nn.Module):
    """경량 ResNet 스타일 모델 (12채널 입력)"""

    def __init__(self, in_channels=12, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())
        self.block1 = self._make_block(64, 64)
        self.block2 = self._make_block(64, 128, stride=2)
        self.block3 = self._make_block(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        return ResBlock(in_ch, out_ch, stride, downsample)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block3(self.block2(self.block1(x)))
        return self.fc(self.pool(x).flatten(1))


for name, M in [("SimpleCNN", SimpleCNN), ("MiniResNet", MiniResNet)]:
    n = sum(p.numel() for p in M(N_BANDS, n_classes).parameters())
    print(f"  {name}: {n:,}개 파라미터")


# ============================================================
# 4. 학습·평가 함수
# ============================================================
def get_device():
    """크로스 플랫폼 디바이스 자동 감지 (CUDA → MPS → CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_and_evaluate(model, train_loader, val_loader, epochs=15, lr=1e-3):
    device = get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.extend(p)
            trues.extend(yb.numpy())
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    acc = np.mean(np.array(preds) == np.array(trues))
    return acc, f1, trues, preds


EPOCHS, BATCH = 15, 32

# ============================================================
# 5. Random Split vs Spatial (타일) Split
# ============================================================
print("\n--- 4. Random Split vs Spatial Split ---")
print("\n  [Random 5-Fold CV]")
r_accs, r_f1s = [], []
for fold, (tr, va) in enumerate(
        KFold(5, shuffle=True, random_state=42).split(range(len(dataset)))):
    tl = DataLoader(Subset(dataset, tr), batch_size=BATCH, shuffle=True)
    vl = DataLoader(Subset(dataset, va), batch_size=BATCH)
    acc, f1, _, _ = train_and_evaluate(SimpleCNN(N_BANDS, n_classes), tl, vl, EPOCHS)
    r_accs.append(acc)
    r_f1s.append(f1)
    print(f"    Fold {fold + 1}: Acc={acc:.3f}, Macro-F1={f1:.3f}")
print(f"    평균: Acc={np.mean(r_accs):.3f}±{np.std(r_accs):.3f}, "
      f"Macro-F1={np.mean(r_f1s):.3f}±{np.std(r_f1s):.3f}")

print("\n  [Spatial CV — 타일 기반]")
b_accs, b_f1s = [], []
gkf = GroupKFold(n_splits=5)
for fold, (tr, va) in enumerate(gkf.split(range(len(dataset)), groups=tile_patch)):
    tl = DataLoader(Subset(dataset, tr), batch_size=BATCH, shuffle=True)
    vl = DataLoader(Subset(dataset, va), batch_size=BATCH)
    acc, f1, _, _ = train_and_evaluate(SimpleCNN(N_BANDS, n_classes), tl, vl, EPOCHS)
    b_accs.append(acc)
    b_f1s.append(f1)
    print(f"    Fold {fold + 1}: Acc={acc:.3f}, Macro-F1={f1:.3f}")
print(f"    평균: Acc={np.mean(b_accs):.3f}±{np.std(b_accs):.3f}, "
      f"Macro-F1={np.mean(b_f1s):.3f}±{np.std(b_f1s):.3f}")

print(f"\n  과대추정 (Random - Spatial):")
print(f"    Accuracy: {np.mean(r_accs) - np.mean(b_accs):+.3f}")
print(f"    Macro-F1: {np.mean(r_f1s) - np.mean(b_f1s):+.3f}")

# ============================================================
# 6. 공간 분할 고정 (train/val) — 모델·밴드 비교용
# ============================================================
# 타일을 학습/검증으로 공간 분리(누수 없는 고정 분할)
utiles = np.unique(tile_patch)
rng.shuffle(utiles)
val_tiles = set(utiles[:len(utiles) // 5])
val_mask = np.array([t in val_tiles for t in tile_patch])
tr_idx = np.where(~val_mask)[0]
va_idx = np.where(val_mask)[0]

# ============================================================
# 7. SimpleCNN vs MiniResNet
# ============================================================
print("\n--- 5. SimpleCNN vs MiniResNet (공간 분할) ---")
tl = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH, shuffle=True)
vl = DataLoader(Subset(dataset, va_idx), batch_size=BATCH)
model_results = {}
for name, M in [("SimpleCNN", SimpleCNN), ("MiniResNet", MiniResNet)]:
    t0 = time.time()
    acc, f1, trues, preds = train_and_evaluate(M(N_BANDS, n_classes), tl, vl, EPOCHS)
    model_results[name] = {"acc": acc, "f1": f1, "time": time.time() - t0}
    print(f"  {name}: Acc={acc:.3f}, Macro-F1={f1:.3f}, 학습시간={model_results[name]['time']:.1f}초")

# ============================================================
# 8. 밴드 선택 비교: RGB(3) vs RGB+NIR(4) vs 전체(12)
# ============================================================
print("\n--- 6. 밴드 선택 비교 (공간 분할) ---")
# 12밴드 순서: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
band_configs = {
    "RGB (3밴드)": [1, 2, 3],          # B02,B03,B04
    "RGB+NIR (4밴드)": [1, 2, 3, 7],   # +B08
    "전체 (12밴드)": list(range(12)),
}
for cfg_name, bidx in band_configs.items():
    sub = PatchDataset(X_patch[:, bidx], y_patch)
    tl = DataLoader(Subset(sub, tr_idx), batch_size=BATCH, shuffle=True)
    vl = DataLoader(Subset(sub, va_idx), batch_size=BATCH)
    acc, f1, _, _ = train_and_evaluate(SimpleCNN(len(bidx), n_classes), tl, vl, EPOCHS)
    print(f"  {cfg_name}: Acc={acc:.3f}, Macro-F1={f1:.3f}")

# ============================================================
# 9. 클래스별 성능 (SimpleCNN 12밴드, 공간 분할)
# ============================================================
print("\n--- 7. 클래스별 성능 (SimpleCNN 12밴드) ---")
tl = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH, shuffle=True)
vl = DataLoader(Subset(dataset, va_idx), batch_size=BATCH)
_, _, trues, preds = train_and_evaluate(SimpleCNN(N_BANDS, n_classes), tl, vl, EPOCHS)
print(classification_report(trues, preds, target_names=class_names,
                            labels=list(range(n_classes)), digits=3, zero_division=0))

# ============================================================
# 10. 요약
# ============================================================
print("--- 8. 요약 ---")
print(f"  {'항목':14s} | {'Random CV':>10s} | {'Spatial CV':>10s} | {'과대추정':>8s}")
print(f"  {'-'*14} | {'-'*10} | {'-'*10} | {'-'*8}")
print(f"  {'Accuracy':14s} | {np.mean(r_accs):>10.3f} | {np.mean(b_accs):>10.3f} | "
      f"{np.mean(r_accs) - np.mean(b_accs):>+8.3f}")
print(f"  {'Macro-F1':14s} | {np.mean(r_f1s):>10.3f} | {np.mean(b_f1s):>10.3f} | "
      f"{np.mean(r_f1s) - np.mean(b_f1s):>+8.3f}")
print("\n  단일 장면(춘천, 2021-04-07) 기반이므로 다른 지역·계절 일반화는")
print("  이 수치로 보장되지 않는다. 4월 초라 경작지가 나지 상태인 점도 라벨과 어긋난다.")

# ============================================================
# 11. 재현성 점검 — 검증 방식의 차이가 실행 간 변동보다 큰가
# ============================================================
# 위에서 나온 '과대추정' 값은 소수점 셋째 자리다. GPU 연산은 완전히 결정적이지
# 않고 가중치 초기화도 시드에 따라 달라지므로, 그 크기의 차이는 실행을 다시 할
# 때마다 흔들릴 수 있다. 차이의 부호를 해석하기 전에 그 흔들림의 폭부터 잰다.
print("\n--- 9. 재현성 점검: 과대추정 차이가 실행 간 변동보다 큰가 ---")
REPEAT_SEEDS = [0, 1, 2]
gap_acc, gap_f1 = [], []
for s in REPEAT_SEEDS:
    torch.manual_seed(s)
    ra, rf, ba, bf = [], [], [], []
    for tr, va in KFold(5, shuffle=True, random_state=s).split(range(len(dataset))):
        a, f, _, _ = train_and_evaluate(
            SimpleCNN(N_BANDS, n_classes),
            DataLoader(Subset(dataset, tr), batch_size=BATCH, shuffle=True),
            DataLoader(Subset(dataset, va), batch_size=BATCH), EPOCHS)
        ra.append(a)
        rf.append(f)
    for tr, va in GroupKFold(n_splits=5).split(range(len(dataset)), groups=tile_patch):
        a, f, _, _ = train_and_evaluate(
            SimpleCNN(N_BANDS, n_classes),
            DataLoader(Subset(dataset, tr), batch_size=BATCH, shuffle=True),
            DataLoader(Subset(dataset, va), batch_size=BATCH), EPOCHS)
        ba.append(a)
        bf.append(f)
    gap_acc.append(np.mean(ra) - np.mean(ba))
    gap_f1.append(np.mean(rf) - np.mean(bf))
    print(f"  시드 {s}: Random Acc={np.mean(ra):.3f} F1={np.mean(rf):.3f} | "
          f"Spatial Acc={np.mean(ba):.3f} F1={np.mean(bf):.3f} | "
          f"과대추정 Acc={gap_acc[-1]:+.3f} F1={gap_f1[-1]:+.3f}")
print(f"  과대추정 Accuracy 범위: {min(gap_acc):+.3f} ~ {max(gap_acc):+.3f}")
print(f"  과대추정 Macro-F1 범위: {min(gap_f1):+.3f} ~ {max(gap_f1):+.3f}")
print("  → 이 범위가 0을 걸치면 '과대추정이 검출되지 않았다'로 읽어야 하고,")
print("    차이의 부호(어느 쪽이 높은가)를 해석해서는 안 된다.")

print("\n[완료] CNN 기반 위성영상 분류 실습을 마쳤다.")
