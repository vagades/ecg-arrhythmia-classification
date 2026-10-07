from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ECGCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_data():
    df = pd.read_csv(BEATS_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]
    X = df[signal_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y


def make_loaders(X_train, X_test, y_train, y_test, batch_size=256):
    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_te = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    y_te = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted")
    return avg_loss, acc, macro_f1, weighted_f1, np.array(all_targets), np.array(all_preds)


def plot_training_curves(history, prefix="cnn"):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    X, y = load_data()
    print("[INFO] Размер X:", X.shape)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print("[INFO] Классы:", class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"[OK] Scaler сохранён: {MODEL_DIR / 'scaler.pkl'}")

    train_loader, test_loader = make_loaders(X_train, X_test, y_train, y_test)

    # Взвешенная функция потерь — улучшает распознавание редкого класса A
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_names)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = ECGCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    epochs = 20
    patience = 5
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_macro_f1 = -1.0
    best_model_path = MODEL_DIR / "best_ecg_cnn.pt"
    no_improve = 0

    print("[INFO] Обучаю CNN...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
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
        val_loss, val_acc, val_macro_f1, _, _, _ = evaluate_model(
            model, test_loader, device, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_macro_f1)

        print(
            f"[EPOCH {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
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

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_macro_f1, test_weighted_f1, y_true, y_pred = evaluate_model(
        model, test_loader, device, criterion
    )

    print("\n[RESULT] CNN test results:")
    print("[RESULT] Accuracy:   ", round(test_acc, 4))
    print("[RESULT] Macro F1:   ", round(test_macro_f1, 4))
    print("[RESULT] Weighted F1:", round(test_weighted_f1, 4))
    print("\n[RESULT] Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — CNN")
    plt.tight_layout()
    out_path = FIG_DIR / "confusion_matrix_cnn.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix сохранена: {out_path}")

    plot_training_curves(history, prefix="cnn")


if __name__ == "__main__":
    main()
