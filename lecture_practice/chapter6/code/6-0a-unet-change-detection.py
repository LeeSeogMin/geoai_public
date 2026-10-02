"""
6-0a. 선택 실습: U-Net으로 두 시점 NDVI 변화 마스크 만들기
===========================================================
기존 6장 실습은 사전 생성 마스크를 CPU에서 채점·집계한다. 이 선택 실습은
PyTorch를 설치한 수강생이 작은 U-Net을 직접 학습해 마스크가 만들어지는
과정을 확인한다. 기존 change_masks.npz와 결과 파일은 수정하지 않는다.

입력  : ndvi_t1, ndvi_t2 (2채널)
정답  : true_change (픽셀별 0/1)
분할  : 32x32 타일 16개 중 오른쪽 아래 4개를 공간 검증 영역으로 고정
장치  : auto는 CUDA -> MPS -> CPU 순으로 선택

실행 예:
    python lecture_practice/chapter6/code/6-0a-unet-change-detection.py
    python lecture_practice/chapter6/code/6-0a-unet-change-detection.py --device cuda
    python lecture_practice/chapter6/code/6-0a-unet-change-detection.py --device cpu --epochs 2

이 코드는 구조 학습용이다. 하나의 합성 장면만 사용하므로 여기서 얻은 성능을
실제 위성영상에 일반화할 수 없다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch가 없습니다. 가상환경을 활성화한 뒤 다음 명령을 실행하세요.\n"
        "python lecture_practice/setup_torch.py"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "change_masks.npz"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
TILE_SIZE = 32
SEED = 20261002


def parse_args():
    parser = argparse.ArgumentParser(
        description="작은 U-Net으로 두 시점 NDVI 변화 마스크를 학습한다."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="학습 장치. auto는 CUDA, MPS, CPU 순으로 고른다.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="학습 반복 수")
    parser.add_argument("--batch-size", type=int, default=16, help="미니배치 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="학습률")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 그림을 저장하지 않는다.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs는 1 이상이어야 합니다.")
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate는 0보다 커야 합니다.")
    return args


def select_device(requested: str) -> torch.device:
    """요청한 장치가 실제 계산 가능한지 확인하고 반환한다."""
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == "cuda" and not cuda_ok:
        raise SystemExit(
            "CUDA를 요청했지만 PyTorch가 CUDA를 사용할 수 없습니다.\n"
            "먼저 확인: python lecture_practice/check_env.py\n"
            "다시 설치: python lecture_practice/setup_torch.py"
        )
    if requested == "mps" and not mps_ok:
        raise SystemExit(
            "MPS를 요청했지만 PyTorch가 Apple GPU를 사용할 수 없습니다.\n"
            "먼저 확인: python lecture_practice/check_env.py\n"
            "다시 설치: python lecture_practice/setup_torch.py"
        )
    if requested != "auto":
        return torch.device(requested)
    if cuda_ok:
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def load_scene():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"데이터가 없습니다: {DATA_PATH}\n"
            "저장소를 다시 내려받아 chapter6/data/change_masks.npz를 확인하세요."
        )
    with np.load(DATA_PATH) as data:
        x = np.stack([data["ndvi_t1"], data["ndvi_t2"]]).astype(np.float32)
        y = data["true_change"].astype(np.float32)[None, ...]
    if x.shape != (2, 128, 128) or y.shape != (1, 128, 128):
        raise SystemExit(f"예상하지 못한 배열 크기입니다: 입력 {x.shape}, 정답 {y.shape}")
    return x, y


def split_tiles(x: np.ndarray, y: np.ndarray):
    """오른쪽 아래 64x64를 검증 영역으로 남기고 나머지를 학습에 쓴다."""
    train, valid = [], []
    for row in range(0, x.shape[1], TILE_SIZE):
        for col in range(0, x.shape[2], TILE_SIZE):
            item = (
                x[:, row : row + TILE_SIZE, col : col + TILE_SIZE],
                y[:, row : row + TILE_SIZE, col : col + TILE_SIZE],
                row,
                col,
            )
            if row >= 64 and col >= 64:
                valid.append(item)
            else:
                train.append(item)
    return train, valid


class TileDataset(Dataset):
    """학습 타일에는 회전·반전을 적용하고 검증 타일은 그대로 둔다."""

    def __init__(self, tiles, augment=False):
        self.tiles = tiles
        self.augment = augment
        self.variants = 8 if augment else 1

    def __len__(self):
        return len(self.tiles) * self.variants

    def __getitem__(self, index):
        tile_index, variant = divmod(index, self.variants)
        x, y, _, _ = self.tiles[tile_index]
        if self.augment:
            turns = variant % 4
            x = np.rot90(x, turns, axes=(1, 2))
            y = np.rot90(y, turns, axes=(1, 2))
            if variant >= 4:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    """32x32 타일용 2단 U-Net. 스킵 연결은 cat 호출 두 곳이다."""

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(2, 16)
        self.enc2 = DoubleConv(16, 32)
        self.bottom = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        x = self.bottom(self.pool(skip2))
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        return self.out(x)


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + smooth) / (total + smooth)).mean()


def binary_metrics(pred: np.ndarray, true: np.ndarray):
    pred, true = pred.astype(bool), true.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    iou = intersection / union if union else 0.0
    denom = pred.sum() + true.sum()
    dice = 2.0 * intersection / denom if denom else 0.0
    return float(iou), float(dice)


def predict_tiles(model, x, tiles, mean, std, device):
    canvas = np.zeros(x.shape[1:], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for tile, _, row, col in tiles:
            normalized = (tile - mean) / std
            xb = torch.from_numpy(normalized[None]).to(device)
            probs = torch.sigmoid(model(xb))[0, 0].cpu().numpy()
            canvas[row : row + TILE_SIZE, col : col + TILE_SIZE] = probs
    return canvas


def save_figure(x, y, probability, valid_mask, device_name):
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"6-0a-unet-{device_name}.png"
    if output.exists():
        print(f"알림: 기존 결과 그림을 덮어씁니다: {output}")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    panels = [x[0], x[1], y[0], probability]
    titles = ["NDVI t1", "NDVI t2", "True change", "U-Net probability"]
    cmaps = ["YlGn", "YlGn", "gray", "magma"]
    for ax, panel, title, cmap in zip(axes, panels, titles, cmaps):
        image = ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_axis_off()
        if title == "U-Net probability":
            ax.contour(valid_mask, levels=[0.5], colors="cyan", linewidths=1)
            fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = select_device(args.device)
    x, y = load_scene()
    train_tiles, valid_tiles = split_tiles(x, y)

    # 정규화 통계와 양성 가중치는 학습 영역에서만 계산한다.
    train_x = np.concatenate([tile[0].reshape(2, -1) for tile in train_tiles], axis=1)
    train_y = np.concatenate([tile[1].reshape(-1) for tile in train_tiles])
    mean = train_x.mean(axis=1).reshape(2, 1, 1).astype(np.float32)
    std = np.maximum(train_x.std(axis=1), 1e-6).reshape(2, 1, 1).astype(np.float32)
    normalized_train = [((tile - mean) / std, mask, row, col)
                        for tile, mask, row, col in train_tiles]

    positives = float(train_y.sum())
    negatives = float(train_y.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)

    loader = DataLoader(
        TileDataset(normalized_train, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("=" * 64)
    print("U-Net 두 시점 NDVI 변화 탐지 (6장 선택 실습)")
    print("=" * 64)
    print(f"장치: {device} | 학습 타일: {len(train_tiles)} | 검증 타일: {len(valid_tiles)}")
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    if device.type == "cpu":
        print("안내: CPU로 실행 중입니다. GPU 실습은 --device cuda 또는 --device mps를 사용하세요.")

    report_every = max(1, args.epochs // 5)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb) + dice_loss(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % report_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}/{args.epochs}: train loss={np.mean(losses):.4f}")

    all_tiles = train_tiles + valid_tiles
    probability = predict_tiles(model, x, all_tiles, mean, std, device)
    prediction = probability >= 0.5
    valid_mask = np.zeros(y.shape[1:], dtype=bool)
    valid_mask[64:, 64:] = True
    iou, dice = binary_metrics(prediction[valid_mask], y[0][valid_mask])
    print(f"검증 영역: IoU={iou:.3f} | Dice={dice:.3f}")
    print("주의: 하나의 합성 장면을 공간 분할한 결과이며 실제 영상 성능이 아닙니다.")

    if not args.no_save:
        output = save_figure(x, y, probability, valid_mask, device.type)
        print(f"결과 그림: {output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"PyTorch 실행 오류: {exc}", file=sys.stderr)
        print("장치 상태 확인: python lecture_practice/check_env.py", file=sys.stderr)
        raise SystemExit(1) from exc
