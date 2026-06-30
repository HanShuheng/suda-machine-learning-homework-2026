"""训练并评估 LSTM、Transformer 与改进模型。

默认实验遵循考核要求：过去 90 天多变量输入，分别预测未来 90 天和 365 天
Global_active_power，并对 5 个随机种子统计 MSE/MAE 均值与标准差。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_COL = "Global_active_power"
INPUT_LEN = 90


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def make_windows(data: np.ndarray, target_idx: int, input_len: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for start in range(len(data) - input_len - horizon + 1):
        mid = start + input_len
        end = mid + horizon
        xs.append(data[start:mid])
        ys.append(data[mid:end, target_idx])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def inverse_target(values: np.ndarray, scaler: MinMaxScaler, target_idx: int, n_features: int) -> np.ndarray:
    tmp = np.zeros((len(values), n_features), dtype=np.float32)
    tmp[:, target_idx] = values
    return scaler.inverse_transform(tmp)[:, target_idx]


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    test_x: torch.Tensor
    test_y: torch.Tensor
    scaler: MinMaxScaler
    target_idx: int
    n_features: int
    feature_names: list[str]


def load_dataset(data_dir: Path, horizon: int, batch_size: int) -> DatasetBundle:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    for df in [train_df, test_df]:
        df.drop(columns=["Date"], inplace=True)

    feature_names = train_df.columns.tolist()
    target_idx = feature_names.index(TARGET_COL)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    train_x, train_y = make_windows(train_scaled, target_idx, INPUT_LEN, horizon)
    test_x, test_y = make_windows(test_scaled, target_idx, INPUT_LEN, horizon)
    if len(train_x) == 0 or len(test_x) == 0:
        raise ValueError(f"数据长度不足，无法构造 horizon={horizon} 的样本")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    return DatasetBundle(
        train_loader=train_loader,
        test_x=torch.tensor(test_x),
        test_y=torch.tensor(test_y),
        scaler=scaler,
        target_idx=target_idx,
        n_features=len(feature_names),
        feature_names=feature_names,
    )


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 96, layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        return self.head(out[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, dim: int = 96, heads: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.pos = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        encoded = self.encoder(z)
        return self.head(encoded[:, -1])


class HybridPyramidForecaster(nn.Module):
    """卷积局部特征 + 金字塔池化 + Transformer 编码的改进模型。"""

    def __init__(self, input_dim: int, horizon: int, dim: int = 96, heads: int = 4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(input_dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.scale2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.scale4 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.fuse = nn.Linear(dim * 3, dim)
        self.pos = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.local(x.transpose(1, 2))
        s2 = torch.nn.functional.interpolate(self.scale2(base), size=base.size(-1), mode="linear")
        s4 = torch.nn.functional.interpolate(self.scale4(base), size=base.size(-1), mode="linear")
        z = torch.cat([base, s2, s4], dim=1).transpose(1, 2)
        z = self.pos(self.fuse(z))
        encoded = self.encoder(z)
        return self.head(encoded[:, -1])


def build_model(name: str, input_dim: int, horizon: int) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(input_dim, horizon)
    if name == "transformer":
        return TransformerForecaster(input_dim, horizon)
    if name == "hybrid":
        return HybridPyramidForecaster(input_dim, horizon)
    raise ValueError(f"未知模型：{name}")


def train_one(model: nn.Module, bundle: DatasetBundle, epochs: int, lr: float, device: torch.device) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in bundle.train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
            count += len(batch_x)
        losses.append(total / max(count, 1))
    return losses


def evaluate(model: nn.Module, bundle: DatasetBundle, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(bundle.test_x), 64):
            pred = model(bundle.test_x[start : start + 64].to(device)).cpu().numpy()
            preds.append(pred)
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = bundle.test_y.numpy()

    pred_real = np.vstack(
        [inverse_target(row, bundle.scaler, bundle.target_idx, bundle.n_features) for row in pred_scaled]
    )
    true_real = np.vstack(
        [inverse_target(row, bundle.scaler, bundle.target_idx, bundle.n_features) for row in true_scaled]
    )
    mse = float(np.mean((pred_real - true_real) ** 2))
    mae = float(np.mean(np.abs(pred_real - true_real)))
    return mse, mae, pred_real, true_real


def plot_prediction(path: Path, pred: np.ndarray, true: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(true, label="Ground Truth", linewidth=2)
    plt.plot(pred, label="Prediction", linewidth=2)
    plt.xlabel("Future day")
    plt.ylabel("Global active power")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_loss(path: Path, losses: list[float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "hybrid"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    all_runs: dict[str, dict] = {}
    for horizon in args.horizons:
        bundle = load_dataset(Path(args.data_dir), horizon, args.batch_size)
        for model_name in args.models:
            for seed in args.seeds:
                set_seed(seed)
                model = build_model(model_name, bundle.n_features, horizon)
                losses = train_one(model, bundle, args.epochs, args.lr, device)
                mse, mae, pred, true = evaluate(model, bundle, device)
                rows.append({"model": model_name, "horizon": horizon, "seed": seed, "mse": mse, "mae": mae})
                key = f"{model_name}_{horizon}_{seed}"
                all_runs[key] = {"mse": mse, "mae": mae, "last_loss": losses[-1]}
                plot_loss(
                    Path(args.figures_dir) / f"{model_name}_{horizon}_{seed}_loss.png",
                    losses,
                    f"{model_name} {horizon}-day training loss",
                )
                plot_prediction(
                    Path(args.figures_dir) / f"{model_name}_{horizon}_{seed}_prediction.png",
                    pred[0],
                    true[0],
                    f"{model_name} {horizon}-day forecast",
                )
                print(f"{model_name} horizon={horizon} seed={seed}: MSE={mse:.4f}, MAE={mae:.4f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(results_dir / "detail.csv", index=False)
    summary = (
        detail.groupby(["model", "horizon"])
        .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .reset_index()
    )
    summary.to_csv(results_dir / "summary.csv", index=False)
    (results_dir / "runs.json").write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
