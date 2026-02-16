import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from model import DualHeadClassifier
print("model Loaded")

#Paths
DATA_PATH = "/scratch/users/k24015724/wav2vec_project/data/IEMOCAP_mean_features.pt"
MODEL_SAVE_PATH = "/scratch/users/k24015724/wav2vec_project/models/dualhead_iemocap.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 768
EMOTION_CLASSES = 5

#class
class IEMOCAPDataset(Dataset):
    def __init__(self, path_to_pt):
        self.data = torch.load(path_to_pt)
        self.label_map = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "fru": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        return features.float(), self.label_map[label]

#Training
def train():
    dataset = IEMOCAPDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    model = DualHeadClassifier(input_dim=FEATURE_DIM, emotion_classes=EMOTION_CLASSES, binary_classes=2)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {len(dataset)} samples for 10 epochs...")
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for features, labels in tqdm(dataloader):
            features, labels = features.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits_emotion, _ = model(features)
            loss = loss_fn(logits_emotion, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
