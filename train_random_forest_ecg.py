from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(BEATS_PATH)
    signal_cols = [col for col in df.columns if col.startswith("x_")]
    X = df[signal_cols].values
    y = df["label"].values
    return X, y


def main():
    X, y = load_data()

    print("[INFO] Размер X:", X.shape)
    print("[INFO] Размер y:", y.shape)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print("[INFO] Классы:", class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print("[INFO] Train shape:", X_train.shape)
    print("[INFO] Test shape:", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    print("[INFO] Обучаю RandomForest...")
    model.fit(X_train, y_train)

    model_path = MODEL_DIR / "random_forest.pkl"
    joblib.dump(model, model_path)
    print(f"[OK] Модель сохранена: {model_path}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n[RESULT] Accuracy:", round(acc, 4))
    print("[RESULT] Macro F1:", round(macro_f1, 4))
    print("[RESULT] Weighted F1:", round(weighted_f1, 4))
    print("\n[RESULT] Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — RandomForest")
    plt.tight_layout()
    output_path = FIG_DIR / "confusion_matrix_random_forest.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix сохранена: {output_path}")

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
    plt.title("RandomForest Confusion Matrix (normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    norm_path = FIG_DIR / "cm_rf_normalized.png"
    plt.savefig(norm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Нормализованная confusion matrix сохранена: {norm_path}")

    # Важность признаков
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]
    plt.figure(figsize=(12, 5))
    plt.bar(range(30), importances[indices])
    plt.xticks(range(30), [f"x_{i}" for i in indices], rotation=90)
    plt.title("RandomForest — топ-30 важных признаков (временные точки ЭКГ)")
    plt.xlabel("Признак")
    plt.ylabel("Важность")
    plt.tight_layout()
    fi_path = FIG_DIR / "rf_feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Важность признаков сохранена: {fi_path}")


if __name__ == "__main__":
    main()
