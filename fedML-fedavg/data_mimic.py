# data_mimic.py

import os
import io
from typing import Tuple, Dict

import pandas as pd
from openpyxl import load_workbook
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


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

_GLOBAL_DF = None  # cache Excel→DataFrame across calls


def _convert_label(x):
    """
    +1 -> 1  (positive)
    0, -1, 2, NaN -> 0  (negative / uncertain / missing)
    """
    if pd.isna(x):
        return 0
    if x == 1:
        return 1
    if x in (0, -1, 2):
        return 0
    return 0


def _extract_images_and_labels(excel_path: str, image_dir: str) -> pd.DataFrame:
    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_excel(excel_path)
    df["image_path"] = None

    wb = load_workbook(excel_path)
    ws = wb.active

    # Map each embedded image object to the corresponding row
    for img in getattr(ws, "_images", []):
        anchor_from = img.anchor._from
        excel_row = anchor_from.row + 1  # Excel is 1-based row index
        df_idx = excel_row - 2           # row 1 header => row 2 is df index 0
        if df_idx < 0 or df_idx >= len(df):
            continue

        raw = img._data()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

        fname = f"cxr_{df_idx:06d}.png"
        fpath = os.path.join(image_dir, fname)
        pil_img.save(fpath)

        df.at[df_idx, "image_path"] = fpath

    # Convert labels to 0/1
    for col in LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"Expected label column '{col}' not in Excel")
        df[col] = df[col].map(_convert_label)

    df = df.dropna(subset=["image_path"])
    return df.reset_index(drop=True)


class MIMICCXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[LABEL_COLS].astype(float).values, dtype=torch.float32)
        return img, labels


def _get_global_df(args) -> pd.DataFrame:
    """Load and cache the full MIMIC dataframe once."""
    global _GLOBAL_DF
    if _GLOBAL_DF is None:
        excel_dir = getattr(args, "data_cache_dir", None)
        if excel_dir is None:
            excel_dir = getattr(args, "data_dir", None) or getattr(args, "dataset_dir", None)
        if excel_dir is None:
            excel_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        excel_path = os.path.join(excel_dir, "mimic-cxr-labeled.xlsx")
        image_dir = os.path.join(excel_dir, "images")
        _GLOBAL_DF = _extract_images_and_labels(excel_path, image_dir)
    return _GLOBAL_DF


def _partition_indices(num_samples: int, client_id: int, num_clients: int):
    """Contiguous split of indices [0, num_samples) across num_clients."""
    base = num_samples // num_clients
    rem = num_samples % num_clients
    start = client_id * base + min(client_id, rem)
    end = start + base + (1 if client_id < rem else 0)
    return list(range(start, end))


def load_partition_data_mimic(args):
    """
    dataset loader.
    """

    df = _get_global_df(args)
    client_num = args.client_num_in_total
    batch_size = args.batch_size

    data_local_num_dict: Dict[int, int] = {}
    train_data_local_dict: Dict[int, DataLoader] = {}
    test_data_local_dict: Dict[int, DataLoader] = {}

    for client_id in range(client_num):
        idx = _partition_indices(len(df), client_id, client_num)
        part_df = df.iloc[idx].reset_index(drop=True)

        # 80/20 train/test split within this client's partition
        n = len(part_df)
        split = int(0.8 * n)
        train_df = part_df.iloc[:split].reset_index(drop=True)
        test_df = part_df.iloc[split:].reset_index(drop=True)

        train_ds = MIMICCXRDataset(train_df)
        test_ds = MIMICCXRDataset(test_df)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        train_data_local_dict[client_id] = train_loader
        test_data_local_dict[client_id] = test_loader
        data_local_num_dict[client_id] = len(train_ds)

    # Global loaders (not used in local cross-silo simulation)
    global_train_ds = ConcatDataset([dl.dataset for dl in train_data_local_dict.values()])
    global_test_ds  = ConcatDataset([dl.dataset for dl in test_data_local_dict.values()])

    train_data_global = DataLoader(global_train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    test_data_global  = DataLoader(global_test_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    train_data_num = sum(data_local_num_dict.values())
    test_data_num = sum(len(dl.dataset) for dl in test_data_local_dict.values())
    class_num = len(LABEL_COLS)

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
