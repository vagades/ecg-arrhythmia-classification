"""
Random Forest с вейвлет-признаками (ДВП db4, 4 уровня).

Обучение на расширенном наборе признаков:
    240 временных точек + 25 вейвлет-признаков = 265 признаков

Сравнивает RF-baseline (только временные) с RF-Wavelet (временные + вейвлет).
Сохраняет лучшую модель в models/random_forest_wavelet.pkl.

Теоретическая основа:
    Захарова Т.В., Шестаков О.В. «Теория вейвлетов и её применение в обработке
    сигналов». — М.: МастерПринт, 2018.

Запуск:
    python train_rf_wavelet.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.wavelet_features import build_combined_matrix, build_wavelet_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Названия 25 вейвлет-признаков для графика важности
WAVELET_FEATURE_NAMES = [
    f"{band}_{stat}"
    for band in ["cD1", "cD2", "cD3", "cD4", "cA4"]
    for stat in ["mean", "std", "energy", "max_abs", "skew"]
]


def load_data():
    df = pd.read_csv(BEATS_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]
    X = df[signal_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y, signal_cols


def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, label):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    print(f"[INFO] Обучаю RF ({label}), признаков: {X_train.shape[1]} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[RESULT] {label}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, y_pred, acc, macro_f1, weighted_f1


def save_confusion_matrix(y_test, y_pred, class_names, filename, title):
    cm = confusion_matrix(y_test, y_pred)
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
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def plot_wavelet_feature_importance(model, raw_dim, feature_names_raw, title, filename):
    """
    Визуализирует важность признаков RF.
    Топ-20 временных точек + столбчатая диаграмма по вейвлет-подполосам.
    """
    importances = model.feature_importances_

    # 1. Суммарная важность по группам (временные vs. каждая вейвлет-подполоса)
    raw_imp = importances[:raw_dim].sum()
    wt_imp_per_band = {}
    for b_idx, band in enumerate(["cD1", "cD2", "cD3", "cD4", "cA4"]):
        start = raw_dim + b_idx * 5
        wt_imp_per_band[band] = importances[start: start + 5].sum()

    groups = ["Временные\n(x_0…x_239)"] + list(wt_imp_per_band.keys())
    values = [raw_imp] + list(wt_imp_per_band.values())
    colors = ["#4C9BE8"] + ["#F4A460", "#E88B4C", "#E8604C", "#6CBF6C", "#B06CBF"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(groups, values, color=colors)
    axes[0].set_title("Суммарная важность групп признаков")
    axes[0].set_ylabel("Суммарная важность")
    axes[0].set_xlabel("Группа признаков")
    axes[0].tick_params(axis="x", labelsize=9)

    # 2. Топ-20 признаков среди вейвлет (все 25 имён)
    wt_importances = importances[raw_dim:]
    top_idx = np.argsort(wt_importances)[::-1][:20]
    top_names = [WAVELET_FEATURE_NAMES[i] for i in top_idx]
    top_vals = wt_importances[top_idx]

    axes[1].barh(range(len(top_idx)), top_vals[::-1], color="#F4A460")
    axes[1].set_yticks(range(len(top_idx)))
    axes[1].set_yticklabels(top_names[::-1], fontsize=9)
    axes[1].set_title("Топ-20 вейвлет-признаков по важности")
    axes[1].set_xlabel("Важность")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def plot_comparison(results, filename):
    """Столбчатая диаграмма сравнения RF-Raw vs RF-Wavelet."""
    labels = list(results.keys())
    accs = [results[k]["accuracy"] for k in labels]
    f1s = [results[k]["macro_f1"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C9BE8")
    ax.bar(x + width / 2, f1s, width, label="Macro F1", color="#F4A460")
    ax.set_ylabel("Метрика")
    ax.set_title("Сравнение RF: временные признаки vs. временные + вейвлет")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.93, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in ax.patches:
        ax.annotate(
            f"{bar.get_height():.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = FIG_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def main():
    print("[INFO] Загружаю данные...")
    X_raw, y, signal_cols = load_data()
    print(f"[INFO] X_raw: {X_raw.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"[INFO] Классы: {class_names}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ── 1. RF-Baseline: только 240 временных точек ──────────────────────────
    model_raw, y_pred_raw, acc_raw, f1_raw, wf1_raw = train_and_evaluate(
        X_train_raw, X_test_raw, y_train, y_test, class_names,
        label="RF-Raw (240 признаков)"
    )

    save_confusion_matrix(
        y_test, y_pred_raw, class_names,
        "cm_rf_raw_normalized.png",
        "RF-Raw — нормализованная матрица ошибок"
    )

    # ── 2. RF-Wavelet: 240 + 25 = 265 признаков ─────────────────────────────
    print("\n[INFO] Вычисляю вейвлет-признаки (train)...")
    X_train_wt = build_combined_matrix(X_train_raw)
    print("[INFO] Вычисляю вейвлет-признаки (test)...")
    X_test_wt = build_combined_matrix(X_test_raw)
    print(f"[INFO] X_wt: {X_train_wt.shape}")

    model_wt, y_pred_wt, acc_wt, f1_wt, wf1_wt = train_and_evaluate(
        X_train_wt, X_test_wt, y_train, y_test, class_names,
        label="RF-Wavelet (265 признаков)"
    )

    save_confusion_matrix(
        y_test, y_pred_wt, class_names,
        "cm_rf_wavelet_normalized.png",
        "RF-Wavelet — нормализованная матрица ошибок"
    )

    plot_wavelet_feature_importance(
        model_wt, raw_dim=X_raw.shape[1], feature_names_raw=signal_cols,
        title="RF-Wavelet: важность групп признаков (ДВП db4, 4 уровня)",
        filename="rf_wavelet_feature_importance.png"
    )

    # ── 3. Сравнение ─────────────────────────────────────────────────────────
    results = {
        "RF-Raw\n(240)": {"accuracy": acc_raw, "macro_f1": f1_raw},
        "RF-Wavelet\n(265)": {"accuracy": acc_wt, "macro_f1": f1_wt},
    }
    plot_comparison(results, "rf_wavelet_vs_raw_comparison.png")

    # ── 4. Сохранение ────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "random_forest_wavelet.pkl"
    joblib.dump(model_wt, model_path)
    print(f"\n[OK] Модель RF-Wavelet сохранена: {model_path}")

    # CSV с итогами
    summary = pd.DataFrame([
        {
            "Модель": "RF-Raw",
            "Признаки": X_train_raw.shape[1],
            "Accuracy": round(acc_raw, 4),
            "Macro F1": round(f1_raw, 4),
            "Weighted F1": round(wf1_raw, 4),
        },
        {
            "Модель": "RF-Wavelet",
            "Признаки": X_train_wt.shape[1],
            "Accuracy": round(acc_wt, 4),
            "Macro F1": round(f1_wt, 4),
            "Weighted F1": round(wf1_wt, 4),
        },
    ])
    csv_path = PROJECT_ROOT / "reports" / "rf_wavelet_comparison.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[OK] Таблица сравнения: {csv_path}")
    print("\n── Итог ──────────────────────────────────────────────────────────")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
