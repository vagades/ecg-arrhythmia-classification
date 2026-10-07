"""
Стратифицированная 5-fold кросс-валидация для всех трёх моделей.

Запуск:
    python cross_validation.py

Результаты сохраняются в:
    reports/cv_results_per_fold.csv   — метрики по каждому фолду
    reports/cv_results_summary.csv    — mean ± std
    reports/figures/models/cv_boxplot.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train_cnn_ecg import ECGCNN
from src.models.resnet1d import ResNet1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
CNN_EPOCHS = 10
RESNET_EPOCHS = 10
BATCH_SIZE = 256


def load_data():
    df = pd.read_csv(BEATS_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]
    X = df[signal_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    class_names = list(le.classes_)
    return X, y, class_names


def make_loader(X, y, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)


def eval_torch(model, loader):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = torch.argmax(model(xb), dim=1).cpu().numpy()
            preds.extend(p)
            true.extend(yb.numpy())
    return accuracy_score(true, preds), f1_score(true, preds, average="macro")


def train_cnn_fold(X_train, y_train, X_val, y_val, class_names):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_names)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    model = ECGCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = make_loader(X_tr, y_train, shuffle=True)
    val_loader = make_loader(X_vl, y_val)

    for _ in range(CNN_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    return eval_torch(model, val_loader)


def train_resnet_fold(X_train, y_train, X_val, y_val, class_names):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_names)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    model = ResNet1D(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = make_loader(X_tr, y_train, shuffle=True)
    val_loader = make_loader(X_vl, y_val)

    for _ in range(RESNET_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    return eval_torch(model, val_loader)


def train_rf_fold(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds), f1_score(y_val, preds, average="macro")


def main():
    print(f"[INFO] Device: {DEVICE}")
    X, y, class_names = load_data()
    print(f"[INFO] X: {X.shape} | Классы: {class_names}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"\n── Fold {fold_idx}/{N_SPLITS} ──────────────────────────")

        print("  [RF]  Обучение RandomForest...")
        rf_acc, rf_f1 = train_rf_fold(X_train, y_train, X_val, y_val)
        print(f"  [RF]  acc={rf_acc:.4f}  macro_f1={rf_f1:.4f}")

        print(f"  [CNN] Обучение CNN ({CNN_EPOCHS} эпох)...")
        cnn_acc, cnn_f1 = train_cnn_fold(X_train, y_train, X_val, y_val, class_names)
        print(f"  [CNN] acc={cnn_acc:.4f}  macro_f1={cnn_f1:.4f}")

        print(f"  [RN]  Обучение ResNet1D ({RESNET_EPOCHS} эпох)...")
        rn_acc, rn_f1 = train_resnet_fold(X_train, y_train, X_val, y_val, class_names)
        print(f"  [RN]  acc={rn_acc:.4f}  macro_f1={rn_f1:.4f}")

        fold_results.append({
            "fold": fold_idx,
            "RF_acc":     rf_acc,   "RF_f1":     rf_f1,
            "CNN_acc":    cnn_acc,  "CNN_f1":    cnn_f1,
            "ResNet_acc": rn_acc,   "ResNet_f1": rn_f1,
        })

    df = pd.DataFrame(fold_results)
    per_fold_path = REPORTS_PATH / "cv_results_per_fold.csv"
    df.to_csv(per_fold_path, index=False)
    print(f"\n[OK] Результаты по фолдам: {per_fold_path}")

    # Сводная таблица mean ± std
    summary_rows = []
    for model_name, acc_col, f1_col in [
        ("RandomForest", "RF_acc",     "RF_f1"),
        ("CNN",          "CNN_acc",    "CNN_f1"),
        ("ResNet1D",     "ResNet_acc", "ResNet_f1"),
    ]:
        summary_rows.append({
            "model":    model_name,
            "acc_mean": df[acc_col].mean(),
            "acc_std":  df[acc_col].std(),
            "f1_mean":  df[f1_col].mean(),
            "f1_std":   df[f1_col].std(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = REPORTS_PATH / "cv_results_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n── Сводная таблица кросс-валидации ──────────────────────────")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<14}  "
            f"Accuracy={row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
            f"MacroF1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
        )

    # Box-plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cols, title in [
        (axes[0], ["RF_acc", "CNN_acc", "ResNet_acc"], "Accuracy"),
        (axes[1], ["RF_f1",  "CNN_f1",  "ResNet_f1"],  "Macro F1"),
    ]:
        data = [df[c].values for c in cols]
        bp = ax.boxplot(data, patch_artist=True)
        colors = ["#4C9BE8", "#F4A460", "#6CBF6C"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticklabels(["RandomForest", "CNN", "ResNet1D"])
        ax.set_ylabel(title)
        ax.set_title(f"{N_SPLITS}-fold CV — {title}")
        ax.grid(True, axis="y", alpha=0.4)

    plt.suptitle("Кросс-валидация: сравнение моделей", fontsize=13)
    plt.tight_layout()
    bp_path = FIG_DIR / "cv_boxplot.png"
    plt.savefig(bp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Box-plot: {bp_path}")
    print("[OK] Кросс-валидация завершена.")


if __name__ == "__main__":
    main()
