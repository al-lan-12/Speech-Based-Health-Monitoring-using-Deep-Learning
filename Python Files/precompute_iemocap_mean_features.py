import os
import torch
import pandas as pd
from tqdm import tqdm

FEATURE_DIR = "/scratch/users/k24015724/wav2vec_project/outputs/IEMOCAP_features"
CSV_PATH = "/scratch/users/k24015724/wav2vec_project/data/IEMOCAP_labels_clean.csv"
OUTPUT_PATH = "/scratch/users/k24015724/wav2vec_project/data/IEMOCAP_mean_features.pt"

df = pd.read_csv(CSV_PATH)
data = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    path, label = row['filename'], row['label']
    features = torch.load(path)

    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    if features.shape[1] != 768:
        print(f"Skipping malformed feature {path}")
        continue

    pooled = features.mean(dim=0)  # [768]
    data.append((pooled, label))

torch.save(data, OUTPUT_PATH)
print(f"Saved mean-pooled IEMOCAP features to {OUTPUT_PATH}")

