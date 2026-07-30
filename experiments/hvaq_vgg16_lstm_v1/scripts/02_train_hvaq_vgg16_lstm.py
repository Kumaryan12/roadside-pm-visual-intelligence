# experiments/hvaq_vgg16_lstm_v1/scripts/02_train_hvaq_vgg16_lstm.py

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# ----------------------------
# Dataset Class
# ----------------------------
class HVAQLSTMDataset(Dataset):
    """T=7 LSTM dataset for HVAQ."""
    def __init__(self, df, t_len=7, image_transform=None, numeric_cols=None):
        self.t_len = t_len
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform
        self.numeric_cols = numeric_cols if numeric_cols is not None else []

        # Group by date+location to create sequences
        self.seq_groups = []
        for (date, loc), g in self.df.groupby(["date", "location_id"]):
            if len(g) >= t_len:
                self.seq_groups.append(g.sort_values("image_time"))

    def __len__(self):
        return sum(len(g) - self.t_len + 1 for g in self.seq_groups)

    def __getitem__(self, idx):
        # find which group the idx falls into
        for g in self.seq_groups:
            n_seq = len(g) - self.t_len + 1
            if idx < n_seq:
                seq_df = g.iloc[idx: idx + self.t_len]
                break
            else:
                idx -= n_seq
        else:
            raise IndexError(idx)

        # load images
        imgs = []
        for img_path in seq_df["image_path"]:
            img = Image.open(img_path).convert("RGB")
            if self.image_transform:
                img = self.image_transform(img)
            imgs.append(img)
        x_img = torch.stack(imgs)  # [T, C, H, W]

        # numeric features
        x_num = None
        if self.numeric_cols:
            x_num = torch.tensor(seq_df[self.numeric_cols].fillna(0).values, dtype=torch.float32)

        y = torch.tensor(seq_df["PM2.5"].values[-1], dtype=torch.float32)  # predict last PM2.5

        return {"x_img": x_img, "x_num": x_num, "y": y}

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["Pearson"] = pearsonr(y_true, y_pred)[0]
        out["Spearman"] = spearmanr(y_true, y_pred)[0]
    else:
        out["Pearson"] = np.nan
        out["Spearman"] = np.nan
    return out

# ----------------------------
# Model
# ----------------------------
class VGG16LSTM(nn.Module):
    def __init__(self, numeric_dim=0, lstm_hidden=128, lstm_layers=1, dropout=0.2):
        super().__init__()
        # VGG16 pretrained
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(vgg.features.children()))
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.flatten = nn.Flatten()
        self.cnn_out_dim = 512*7*7

        self.numeric_dim = numeric_dim
        self.lstm_input_dim = self.cnn_out_dim + numeric_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x_img, x_num=None):
        B, T, C, H, W = x_img.shape
        cnn_feats = []
        for t in range(T):
            f = self.cnn(x_img[:,t])
            f = self.pool(f)
            f = self.flatten(f)
            cnn_feats.append(f)
        x_seq = torch.stack(cnn_feats, dim=1)  # [B, T, cnn_out]

        if x_num is not None:
            x_seq = torch.cat([x_seq, x_num], dim=-1)

        lstm_out, _ = self.lstm(x_seq)
        y = self.head(lstm_out[:,-1])
        return y.squeeze(-1)

# ----------------------------
# Training function
# ----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, leave=False):
        x_img = batch["x_img"].to(device)
        x_num = batch["x_num"].to(device) if batch["x_num"] is not None else None
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred = model(x_img, x_num)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

@torch.inference_mode()
def eval_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        x_img = batch["x_img"].to(device)
        x_num = batch["x_num"].to(device) if batch["x_num"] is not None else None
        y = batch["y"].cpu().numpy()
        pred = model(batch["x_img"].to(device), x_num).cpu().numpy()
        y_true.extend(y)
        y_pred.extend(pred)
    return np.array(y_true), np.array(y_pred)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="experiments/hvaq_vgg16_lstm_v1/data/processed/hvaq_image_label_manifest.csv")
    parser.add_argument("--numeric-cols", nargs="+", default=["Temperature","Humidity"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="experiments/hvaq_vgg16_lstm_v1/models/vgg16_lstm")
    parser.add_argument("--fig-dir", default="experiments/hvaq_vgg16_lstm_v1/figures/vgg16_lstm")
    parser.add_argument("--t-len", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.manifest)
    df = df.sort_values(["date","location_id","image_time"]).reset_index(drop=True)

    # simple train/val/test split
    train_df = df[df["date"].isin(df["date"].unique()[:2])].copy()
    val_df = df[df["date"].isin(df["date"].unique()[2:3])].copy()
    test_df = val_df.copy()

    # image transform
    img_trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = HVAQLSTMDataset(train_df, t_len=args.t_len, image_transform=img_trans, numeric_cols=args.numeric_cols)
    val_ds = HVAQLSTMDataset(val_df, t_len=args.t_len, image_transform=img_trans, numeric_cols=args.numeric_cols)
    test_ds = HVAQLSTMDataset(test_df, t_len=args.t_len, image_transform=img_trans, numeric_cols=args.numeric_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VGG16LSTM(numeric_dim=len(args.numeric_cols))
    model.to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_rmse = float("inf")
    best_model_path = out_dir / "best_vgg16_lstm.pt"

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_val, y_val_pred = eval_model(model, val_loader, device)
        y_test, y_test_pred = eval_model(model, test_loader, device)
        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        print(f"[Epoch {epoch}] loss={loss:.4f} | VAL RMSE={val_metrics['RMSE']:.3f}, TEST RMSE={test_metrics['RMSE']:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            **{f"val_{k}": v for k,v in val_metrics.items()},
            **{f"test_{k}": v for k,v in test_metrics.items()},
        })

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(model.state_dict(), best_model_path)

    # save history and predictions
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(fig_dir / "training_history.csv", index=False)

    pred_df = pd.DataFrame({
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred
    })
    pred_df.to_csv(fig_dir / "predictions.csv", index=False)

    print("Saved best model to:", best_model_path)
    print("Saved predictions and history to:", fig_dir)

if __name__ == "__main__":
    main()