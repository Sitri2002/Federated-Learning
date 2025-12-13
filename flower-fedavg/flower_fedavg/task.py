import os
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

import pandas as pd
from openpyxl import load_workbook
from PIL import Image


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
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
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
    """ResNet50 for MIMIC-CXR (224x224 images, multi-label)."""

    def __init__(self, block, layers, num_classes=14):
        super().__init__()
        self.in_channels = 64

        # Initial conv + maxpool (ImageNet-style)
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])          # 3 blocks
        self.layer2 = self._make_layer(block, 128, layers[1], 2)      # 4 blocks
        self.layer3 = self._make_layer(block, 256, layers[2], 2)      # 6 blocks
        self.layer4 = self._make_layer(block, 512, layers[3], 2)      # 3 blocks

        # Pool + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



LABEL_COLS = [
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

# Number of output labels for the model
NUM_CLASSES = len(LABEL_COLS)


def _convert_label(x):
    """Map CheXbert-style labels to simple binary (0/1).

    +1 -> 1  (positive)
     0 -> 0  (negative)
    -1 -> 0  (uncertain, treated as negative)
     2 or NaN -> 0  (missing / not mentioned, treated as negative)
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


def extract_images_and_labels(excel_path: str, image_dir: str) -> pd.DataFrame:
    """Read Excel, extract embedded images, return df with image_path + 14 labels."""

    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_excel(excel_path)
    df["image_path"] = None

    wb = load_workbook(excel_path)
    ws = wb.active

    # Map each embedded image to a row via its anchor
    for img in getattr(ws, "_images", []):
        anchor_from = img.anchor._from
        excel_row = anchor_from.row + 1  # 1-based
        df_idx = excel_row - 2           # row 1 header, row 2 -> index 0
        if df_idx < 0 or df_idx >= len(df):
            continue

        raw = img._data()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

        fname = f"cxr_{df_idx:06d}.png"
        fpath = os.path.join(image_dir, fname)
        pil_img.save(fpath)

        df.at[df_idx, "image_path"] = fpath

    # Convert labels
    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"Label column '{col}' not found in Excel file")
        df[col] = df[col].map(_convert_label)

    # Only require image_path (labels now all 0/1)
    df = df.dropna(subset=["image_path"])
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


# Cache the full dataframe so each client doesn't re-parse Excel repeatedly
_GLOBAL_DF = None


def _get_global_df(excel_path: str, image_dir: str) -> pd.DataFrame:
    global _GLOBAL_DF
    if _GLOBAL_DF is None:
        _GLOBAL_DF = extract_images_and_labels(excel_path, image_dir)
    return _GLOBAL_DF


def _partition_indices(num_samples: int, partition_id: int, num_partitions: int):
    """Split [0, num_samples) into num_partitions contiguous chunks."""
    base = num_samples // num_partitions
    rem = num_samples % num_partitions
    # First `rem` partitions get one extra
    start = partition_id * base + min(partition_id, rem)
    end = start + base + (1 if partition_id < rem else 0)
    return list(range(start, end))


def load_data(partition_id: int, num_partitions: int):
    """Load local MIMIC-CXR data for this client partition.

    Returns: (trainloader, testloader)
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    excel_path = os.path.join(DATA_DIR, "mimic-cxr-labeled.xlsx")
    image_dir = os.path.join(DATA_DIR, "extracted_images")

    df = _get_global_df(excel_path, image_dir)

    idx = _partition_indices(len(df), partition_id, num_partitions)
    part_df = df.iloc[idx].reset_index(drop=True)

    # 80/20 train/test split within this partition
    n = len(part_df)
    split = int(0.8 * n)
    train_df = part_df.iloc[:split].reset_index(drop=True)
    test_df = part_df.iloc[split:].reset_index(drop=True)

    train_ds = MIMICCXRDataset(train_df)
    test_ds = MIMICCXRDataset(test_df)

    trainloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    testloader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    return trainloader, testloader

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set (multi-label BCE)."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / max(len(trainloader), 1)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set (multi-label accuracy)."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = net(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).float().sum().item()
            total += labels.numel()
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / max(len(testloader), 1)
    return avg_loss, accuracy
