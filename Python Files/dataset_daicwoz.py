from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd


class DAICWOZBinaryDataset(Dataset):
    def __init__(self, label_csv: str, feature_dir: str):
        self.label_csv = pd.read_csv(label_csv)
        self.feature_dir = Path(feature_dir)

        if self.label_csv.columns[0] == "Participant_ID":
            self.label_csv = self.label_csv[1:]

        self.labels = {
            int(row["Participant_ID"]): int(row["PHQ8_Binary"])
            for _, row in self.label_csv.iterrows()
        }

        self.available_ids = [
            int(f.stem.split("_")[0])
            for f in self.feature_dir.glob("*.pt")
            if f.is_file()
        ]

        self.valid_ids = [pid for pid in self.labels if pid in self.available_ids]

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_ids[idx]
        feature_path = self.feature_dir / f"{pid}.pt"

        features = torch.load(feature_path)

        if features.dim() == 3:
            features = features.mean(dim=1) 
        elif features.dim() == 2:
            features = features.mean(dim=0) 

        label = torch.tensor([self.labels[pid]], dtype=torch.float32)
        return features, label

