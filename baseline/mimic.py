import argparse
import os
import random
from typing import Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

import pandas as pd
from openpyxl import load_workbook
from PIL import Image
import io


def setup_logger(log_path: str = "train.log"):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # file handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


class Bottleneck(nn.Module):
    expansion = 4  # output channels are 4× the base width

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Stage 1
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 2–5
        self.layer1 = self._make_layer(block, 64, layers[0])   # 3 blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4 blocks
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6 blocks
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3 blocks

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # If we change resolution (stride>1) or #channels, we need projection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        # Stages 2–5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----- MIMIC-CXR specific dataset helpers -----

LABEL_COLS: List[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def _convert_label(x):
    """
    +1 -> 1  (positive)
     0 -> 0  (negative)
    -1 -> 0  (uncertain treated as negative)
     2 or NaN -> 0  (not mentioned / missing treated as negative)
    """
    if pd.isna(x):
        return 0
    if x == 1:
        return 1
    if x in (0, -1):
        return 0
    if x == 2:
        return 0
    return 0


def extract_images_and_labels(excel_path: str, image_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """Read the Excel file, extract embedded images, return a DataFrame with image_path + labels."""
    os.makedirs(image_dir, exist_ok=True)

    logger.info(f"Reading Excel labels from {excel_path}")
    df = pd.read_excel(excel_path)

    # Add image_path column
    df["image_path"] = None

    logger.info("Loading workbook via openpyxl to extract embedded images...")
    wb = load_workbook(excel_path)
    ws = wb.active

    mapped = 0
    for img in getattr(ws, "_images", []):
        # img.anchor._from has zero-based row/col indices
        anchor_from = img.anchor._from
        excel_row = anchor_from.row + 1  # convert to Excel 1-based
        # In your sheet, row 1 is header, row 2 → df index 0, etc.
        df_idx = excel_row - 2
        if df_idx < 0 or df_idx >= len(df):
            continue

        raw = img._data()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

        fname = f"cxr_{df_idx:06d}.png"
        fpath = os.path.join(image_dir, fname)
        pil_img.save(fpath)

        df.at[df_idx, "image_path"] = fpath
        mapped += 1

    logger.info(f"Mapped {mapped} embedded images to rows")

    # Convert labels
    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"Expected label column '{col}' not found in Excel file")

    for col in LABEL_COLS:
        df[col] = df[col].map(_convert_label)
    before = len(df)
    # Only require image_path to be present; labels are now 0/1 everywhere
    df = df.dropna(subset=["image_path"])
    after = len(df)

    logger.info(f"Filtered rows with missing data: {before} -> {after}")

    return df.reset_index(drop=True)


class MIMICCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[LABEL_COLS].astype(float).values, dtype=torch.float32)
        return img, labels


def get_dataloaders(
    excel_path: str,
    image_dir: str,
    batch_size: int,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    logger: logging.Logger = None,
) -> Tuple[DataLoader, DataLoader]:
    if logger is None:
        logger = logging.getLogger()

    df = extract_images_and_labels(excel_path, image_dir, logger)
    print(df)
    # Train/val split
    indices = list(range(len(df)))
    random.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = MIMICCXRDataset(train_df)
    val_ds = MIMICCXRDataset(val_df)

    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, valloader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Multi-label accuracy: per-label correctness after sigmoid+0.5
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).float().sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(num_classes: int) -> nn.Module:
    return ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR ResNet-50 (Excel-embedded images)")
    parser.add_argument("--excel_path", type=str, default = "mimic-cxr-labeled.xlsx",
                        help="Path to the .xlsx file containing embedded images and labels")
    parser.add_argument("--image_dir", type=str, default="extracted_images",
                        help="Directory to save extracted PNG images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="resnet50_mimic.pt")
    parser.add_argument("--log_path", type=str, default="train_mimic.log")
    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.dirname(args.log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(args.log_path)
    logger.info("===== MIMIC-CXR ResNet-50 (Excel-embedded images) =====")
    logger.info(f"Args: {args}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    trainloader, valloader = get_dataloaders(
        args.excel_path,
        args.image_dir,
        args.batch_size,
        args.num_workers,
        train_ratio=0.8,
        logger=logger,
    )

    num_classes = len(LABEL_COLS)
    model = make_model(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valloader, criterion, device)
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}% (per-label)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                args.save_path,
            )

    logger.info(f"Model improved to {best_acc*100:.2f}% → saved to {args.save_path}")
    logger.info(f"Training complete. Best accuracy {best_acc*100:.2f}% (per-label)")
    logging.shutdown()


if __name__ == "__main__":
    main()
