import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from model import DualHeadClassifier
import sys
sys.path.append("/scratch/users/k24015724/wav2vec_project/scripts")
from dataset_daicwoz import DAICWOZBinaryDataset
from torch.utils.data import DataLoader

#Config
FEATURE_DIM = 768
EMOTION_CLASSES = 4
BINARY_CLASSES = 1
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/scratch/users/k24015724/wav2vec_project/models/dualhead_iemocap_finetuned.pt"
label_csv = "/scratch/users/k24015724/wav2vec_project/data/DAICWOZ_meta/dev_split_Depression_AVEC2017.csv"
feature_dir = "/scratch/users/k24015724/wav2vec_project/features/DAICWOZ"

print("model.py Loaded")
print("Starting binary inference on DAIC-WOZ...")

#Metadata
meta = pd.read_csv(label_csv)
meta = meta[meta["PHQ8_Binary"].isin([0, 1])]
meta = meta[meta["Participant_ID"] != "PHQ8_Binary"]  # Remove header duplicates
meta["Participant_ID"] = pd.to_numeric(meta["Participant_ID"], errors='coerce')
meta = meta.dropna(subset=["Participant_ID"])
meta["Participant_ID"] = meta["Participant_ID"].astype(int)

num_pos = (meta["PHQ8_Binary"] == 1).sum()
num_neg = (meta["PHQ8_Binary"] == 0).sum()
if num_pos == 0:
    raise ValueError("No positive samples found in metadata.")
pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE)

#Model
model = DualHeadClassifier(input_dim=FEATURE_DIM, emotion_classes=EMOTION_CLASSES, binary_classes=BINARY_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

#Dataset
dataset = DAICWOZBinaryDataset(label_csv, feature_dir)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

#Inference
all_labels, all_preds = [], []
missing = 0
for features, labels in tqdm(dataloader):
    features, labels = features.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
        _, logits_binary = model(features)
        preds = (torch.sigmoid(logits_binary) > 0.5).int().view(-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.view(-1).cpu().tolist())

#Metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Non-Depressed", "Depressed"]))

