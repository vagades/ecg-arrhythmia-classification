from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from train_cnn_ecg import ECGCNN
from src.models.resnet1d import ResNet1D
from src.preprocessing.add_noise import (
    add_gaussian_noise,
    add_baseline_wander,
    add_powerline_noise,
    add_muscle_noise,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIG_PATH = REPORTS_PATH / "figures" / "models"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)

NOISE_CONFIGS = {
    "gaussian":  [0.01, 0.03, 0.05, 0.08],
    "baseline":  [0.03, 0.06, 0.10, 0.15],
    "powerline": [0.02, 0.04, 0.06, 0.08],
    "muscle":    [0.01, 0.03, 0.05, 0.07],
}

NOISE_LABELS = {
    "gaussian":  "Гауссов шум",
    "baseline":  "Дрейф изолинии",
    "powerline": "Сетевая помеха (50 Гц)",
    "muscle":    "Мышечный шум (ЭМГ)",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]
    X = df[signal_cols].values.astype(np.float32)
    y = df["label"].astype("category").cat.codes.values
    class_names = df["label"].astype("category").cat.categories.tolist()
    return X, y, class_names


def apply_noise(X, noise_type, level):
    X_noisy = np.empty_like(X)
    for i in range(X.shape[0]):
        sig = X[i]
        if noise_type == "gaussian":
            X_noisy[i] = add_gaussian_noise(sig, sigma=level, random_state=42 + i)
        elif noise_type == "baseline":
            X_noisy[i] = add_baseline_wander(sig, amplitude=level)
        elif noise_type == "powerline":
            X_noisy[i] = add_powerline_noise(sig, amplitude=level)
        elif noise_type == "muscle":
            X_noisy[i] = add_muscle_noise(sig, amplitude=level, random_state=42 + i)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    return X_noisy


def eval_cnn(model, scaler, X_noisy, y_test):
    X_scaled = scaler.transform(X_noisy)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_t), dim=1).cpu().numpy()
    return (
        accuracy_score(y_test, preds),
        f1_score(y_test, preds, average="macro"),
        f1_score(y_test, preds, average="weighted"),
    )


def eval_resnet(model, scaler, X_noisy, y_test):
    X_scaled = scaler.transform(X_noisy)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_t), dim=1).cpu().numpy()
    return (
        accuracy_score(y_test, preds),
        f1_score(y_test, preds, average="macro"),
        f1_score(y_test, preds, average="weighted"),
    )


def eval_rf(model, X_noisy, y_test):
    preds = model.predict(X_noisy)
    return (
        accuracy_score(y_test, preds),
        f1_score(y_test, preds, average="macro"),
        f1_score(y_test, preds, average="weighted"),
    )


def plot_noise_type(noise_df, noise_type, metric="macro_f1"):
    plt.figure(figsize=(8, 5))
    for model_name in noise_df["model"].unique():
        subset = noise_df[noise_df["model"] == model_name]
        plt.plot(subset["level"], subset[metric], marker="o", label=model_name)
    plt.xlabel("Уровень шума")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{NOISE_LABELS[noise_type]} — {metric}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = FIG_PATH / f"noise_{noise_type}_{metric}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined(all_df, metric="macro_f1"):
    noise_types = list(NOISE_CONFIGS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, noise_type in zip(axes, noise_types):
        subset_all = all_df[all_df["noise_type"] == noise_type]
        for model_name in subset_all["model"].unique():
            subset = subset_all[subset_all["model"] == model_name]
            ax.plot(subset["level"], subset[metric], marker="o", label=model_name)
        ax.set_title(NOISE_LABELS[noise_type])
        ax.set_xlabel("Уровень шума")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle(f"Устойчивость моделей к шуму — {metric}", fontsize=13)
    plt.tight_layout()
    out = FIG_PATH / f"noise_robustness_{metric}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Сводный график ({metric}) сохранён: {out}")


def main():
    X, y, class_names = load_data()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Загружаем сохранённые scalers и модели
    scaler_cnn = joblib.load(MODELS_PATH / "scaler.pkl")
    scaler_resnet = joblib.load(MODELS_PATH / "scaler_resnet.pkl")

    cnn = ECGCNN(num_classes=len(class_names)).to(DEVICE)
    cnn.load_state_dict(torch.load(MODELS_PATH / "best_ecg_cnn.pt", map_location=DEVICE))

    resnet = ResNet1D(num_classes=len(class_names)).to(DEVICE)
    resnet.load_state_dict(torch.load(MODELS_PATH / "resnet1d_best.pt", map_location=DEVICE))

    rf = joblib.load(MODELS_PATH / "random_forest.pkl")

    all_results = []

    for noise_type, levels in NOISE_CONFIGS.items():
        print(f"\n=== Тип шума: {NOISE_LABELS[noise_type]} ===")
        noise_results = []

        for level in levels:
            X_noisy = apply_noise(X_test, noise_type, level)

            cnn_acc, cnn_macro, cnn_w = eval_cnn(cnn, scaler_cnn, X_noisy, y_test)
            res_acc, res_macro, res_w = eval_resnet(resnet, scaler_resnet, X_noisy, y_test)
            rf_acc, rf_macro, rf_w = eval_rf(rf, X_noisy, y_test)

            for model_name, acc, macro, weighted in [
                ("CNN",          cnn_acc, cnn_macro, cnn_w),
                ("ResNet1D",     res_acc, res_macro,  res_w),
                ("RandomForest", rf_acc,  rf_macro,   rf_w),
            ]:
                row = {
                    "noise_type": noise_type,
                    "level": level,
                    "model": model_name,
                    "accuracy": acc,
                    "macro_f1": macro,
                    "weighted_f1": weighted,
                }
                noise_results.append(row)
                all_results.append(row)

            print(
                f"  level={level:.3f} | "
                f"CNN={cnn_macro:.4f} | "
                f"ResNet={res_macro:.4f} | "
                f"RF={rf_macro:.4f}"
            )

        noise_df = pd.DataFrame(noise_results)

        # Сохраняем CSV и графики для каждого типа шума отдельно
        csv_path = REPORTS_PATH / f"noise_results_{noise_type}.csv"
        noise_df.to_csv(csv_path, index=False)

        plot_noise_type(noise_df, noise_type, metric="accuracy")
        plot_noise_type(noise_df, noise_type, metric="macro_f1")

    all_df = pd.DataFrame(all_results)
    all_df.to_csv(REPORTS_PATH / "noise_results_all.csv", index=False)

    # Сводные графики 2×2 по всем типам шума
    plot_combined(all_df, metric="accuracy")
    plot_combined(all_df, metric="macro_f1")

    print("\n[OK] Все эксперименты завершены.")
    print(f"[OK] CSV-таблицы сохранены в: {REPORTS_PATH}")
    print(f"[OK] Графики сохранены в: {FIG_PATH}")


if __name__ == "__main__":
    main()
