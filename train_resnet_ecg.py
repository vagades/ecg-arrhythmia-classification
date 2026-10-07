import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from src.models.resnet1d import ResNet1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 240) float32 — уже нормализован и будет unsqueeze внутри
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 240)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, X_test, y_train, y_test, batch_size=512):
    train_loader = DataLoader(
        ECGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        ECGDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    weighted_f1 = f1_score(all_true, all_preds, average="weighted")
    return avg_loss, acc, macro_f1, weighted_f1, np.array(all_true), np.array(all_preds)


def plot_training_curves(history, prefix="resnet"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix.upper()} — кривая потерь")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"{prefix}_loss_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] График loss сохранён: {out}")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix.upper()} — кривая точности")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"{prefix}_accuracy_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] График accuracy сохранён: {out}")


def train_resnet(csv_path=DEFAULT_CSV_PATH, epochs=30, batch_size=512, patience=7):
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Загружаю данные из: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = pd.read_csv(csv_path)
    signal_cols = [col for col in df.columns if col.startswith("x_")]

    X = df[signal_cols].values.astype(np.float32)   # (N, 240)
    y = df["label"].astype("category").cat.codes.values
    class_names = df["label"].astype("category").cat.categories.tolist()

    print("[INFO] Размер X:", X.shape, "| Классы:", class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Нормализация — важно для сходимости ResNet
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler_resnet.pkl")
    print(f"[OK] Scaler сохранён: {MODELS_DIR / 'scaler_resnet.pkl'}")

    train_loader, test_loader = make_loaders(X_train, X_test, y_train, y_test, batch_size)

    model = ResNet1D(num_classes=len(class_names)).to(DEVICE)

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_names)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_macro_f1 = -1.0
    best_model_path = MODELS_DIR / "resnet1d_best.pt"
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"[INFO] Обучаю ResNet1D (epochs={epochs}, patience={patience})...\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(yb.detach().cpu().numpy())

        train_loss = total_train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        val_loss, val_acc, val_macro_f1, val_weighted_f1, _, _ = evaluate_model(
            model, test_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_macro_f1)
        epoch_time = time.time() - start_time

        print(
            f"[EPOCH {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] Early stopping на эпохе {epoch} (patience={patience})")
                break

    print(f"\n[OK] Лучшая модель сохранена: {best_model_path}")
    print(f"[OK] Лучший val macro F1: {best_val_macro_f1:.4f}")

    plot_training_curves(history, prefix="resnet")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc, test_macro_f1, test_weighted_f1, y_true, y_pred = evaluate_model(
        model, test_loader, criterion
    )

    print("\n[RESULT] ResNet1D test results:")
    print("[RESULT] Accuracy:   ", round(test_acc, 4))
    print("[RESULT] Macro F1:   ", round(test_macro_f1, 4))
    print("[RESULT] Weighted F1:", round(test_weighted_f1, 4))
    print("\n[RESULT] Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("[RESULT] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    train_resnet()
