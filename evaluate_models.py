import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet1d import ResNet1D
from train_cnn_ecg import ECGCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
MODEL_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports" / "figures" / "models"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


def make_loader(X, y, batch_size=512):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)


def evaluate_torch(model, loader):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            outputs = model(xb)
            p = torch.argmax(outputs, dim=1)
            preds.extend(p.cpu().numpy())
            true.extend(yb.numpy())

    acc = accuracy_score(true, preds)
    macro_f1 = f1_score(true, preds, average="macro")
    weighted_f1 = f1_score(true, preds, average="weighted")
    cm = confusion_matrix(true, preds)
    return acc, macro_f1, weighted_f1, cm, np.array(true), np.array(preds)


def plot_cm(cm, class_names, title, filename):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {filename} сохранена")


def plot_model_comparison(results):
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[m][metric] for m in models]
        plt.bar(x + (i - 1) * width, values, width, label=label)

    plt.xticks(x, models)
    plt.ylim(0.75, 1.0)
    plt.legend()
    plt.title("Сравнение моделей на чистых данных")
    plt.tight_layout()
    path = REPORTS_PATH / "model_comparison_clean.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] График сравнения сохранён: {path}")


def main():
    df = pd.read_csv(DATA_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]

    X = df[signal_cols].values.astype(np.float32)
    y = df["label"].astype("category").cat.codes.values
    class_names = df["label"].astype("category").cat.categories.tolist()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ── CNN ──────────────────────────────────────────────────────────────────
    scaler_cnn = joblib.load(MODEL_PATH / "scaler.pkl")
    X_test_cnn = scaler_cnn.transform(X_test)
    cnn_loader = make_loader(X_test_cnn, y_test)

    cnn = ECGCNN(num_classes=len(class_names)).to(DEVICE)
    cnn.load_state_dict(torch.load(MODEL_PATH / "best_ecg_cnn.pt", map_location=DEVICE))

    cnn_acc, cnn_macro, cnn_weighted, cnn_cm, cnn_true, cnn_pred = evaluate_torch(cnn, cnn_loader)

    print("\n=== CNN ===")
    print(f"Accuracy:    {cnn_acc:.4f}")
    print(f"Macro F1:    {cnn_macro:.4f}")
    print(f"Weighted F1: {cnn_weighted:.4f}")
    print(classification_report(cnn_true, cnn_pred, target_names=class_names))
    plot_cm(cnn_cm, class_names, "CNN Confusion Matrix (normalized)", "cm_cnn_normalized.png")

    results["CNN"] = {"accuracy": cnn_acc, "macro_f1": cnn_macro, "weighted_f1": cnn_weighted}

    # ── ResNet1D ──────────────────────────────────────────────────────────────
    scaler_resnet = joblib.load(MODEL_PATH / "scaler_resnet.pkl")
    X_test_resnet = scaler_resnet.transform(X_test)
    resnet_loader = make_loader(X_test_resnet, y_test)

    resnet = ResNet1D(num_classes=len(class_names)).to(DEVICE)
    resnet.load_state_dict(torch.load(MODEL_PATH / "resnet1d_best.pt", map_location=DEVICE))

    res_acc, res_macro, res_weighted, res_cm, res_true, res_pred = evaluate_torch(resnet, resnet_loader)

    print("\n=== ResNet1D ===")
    print(f"Accuracy:    {res_acc:.4f}")
    print(f"Macro F1:    {res_macro:.4f}")
    print(f"Weighted F1: {res_weighted:.4f}")
    print(classification_report(res_true, res_pred, target_names=class_names))
    plot_cm(res_cm, class_names, "ResNet1D Confusion Matrix (normalized)", "cm_resnet_normalized.png")

    results["ResNet1D"] = {"accuracy": res_acc, "macro_f1": res_macro, "weighted_f1": res_weighted}

    # ── RandomForest ──────────────────────────────────────────────────────────
    rf = joblib.load(MODEL_PATH / "random_forest.pkl")
    rf_pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_macro = f1_score(y_test, rf_pred, average="macro")
    rf_weighted = f1_score(y_test, rf_pred, average="weighted")
    rf_cm = confusion_matrix(y_test, rf_pred)

    print("\n=== RandomForest ===")
    print(f"Accuracy:    {rf_acc:.4f}")
    print(f"Macro F1:    {rf_macro:.4f}")
    print(f"Weighted F1: {rf_weighted:.4f}")
    print(classification_report(y_test, rf_pred, target_names=class_names))
    plot_cm(rf_cm, class_names, "RandomForest Confusion Matrix (normalized)", "cm_rf_normalized.png")

    results["RandomForest"] = {"accuracy": rf_acc, "macro_f1": rf_macro, "weighted_f1": rf_weighted}

    # ── Сводный график ────────────────────────────────────────────────────────
    plot_model_comparison(results)

    # ── Сводная таблица в CSV ─────────────────────────────────────────────────
    rows = [{"model": m, **v} for m, v in results.items()]
    summary = pd.DataFrame(rows)
    summary_path = PROJECT_ROOT / "reports" / "clean_data_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Сводная таблица сохранена: {summary_path}")
    print("\n" + summary.to_string(index=False))
    print(f"\n[OK] Все графики сохранены в: {REPORTS_PATH}")


if __name__ == "__main__":
    main()
