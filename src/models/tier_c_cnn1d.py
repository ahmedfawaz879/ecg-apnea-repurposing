"""Extracted unchanged from notebook Sec. 6 ("Tier C -- Raw ECG Waveform,
1D-CNN, No Hand-Crafted Features"). get_cnn_predictions (Sec. 8) is kept
here too, as the Tier-C-specific inference helper.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.config import Config


class ECGSegmentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_len: int):
        self.segments = df["center_minute_segment"].values
        self.labels = df["label"].values
        self.target_len = target_len

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx].astype(np.float32)
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)
        if len(seg) != self.target_len:
            seg = np.resize(seg, self.target_len)   # defensive: pad/trim to a fixed length
        return {"segment": torch.tensor(seg).unsqueeze(0), "label": torch.tensor(self.labels[idx], dtype=torch.float32)}


class ECG1DCNN(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x).squeeze(-1)
        return self.classifier(feats).squeeze(-1)


def train_tier_c(train_features: pd.DataFrame, cfg: Config, device: torch.device) -> ECG1DCNN:
    tier_c_model = ECG1DCNN(cfg.cnn_hidden).to(device)
    train_ds = ECGSegmentDataset(train_features, cfg.minute_samples)
    train_loader_cnn = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    optim = torch.optim.AdamW(tier_c_model.parameters(), lr=cfg.lr_tier_c)
    tier_c_model.train()
    for epoch in range(cfg.epochs_tier_c):
        epoch_loss, n = 0.0, 0
        for batch in train_loader_cnn:
            seg = batch["segment"].to(device)
            label = batch["label"].to(device)
            optim.zero_grad()
            logits = tier_c_model(seg)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * seg.size(0); n += seg.size(0)
        print(f"[Tier C] epoch {epoch+1}/{cfg.epochs_tier_c}  loss={epoch_loss/max(n,1):.4f}")
    tier_c_model.eval()
    return tier_c_model


@torch.no_grad()
def get_cnn_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        seg = batch["segment"].to(device)
        probs.append(torch.sigmoid(model(seg)).cpu().numpy())
    return np.concatenate(probs)
