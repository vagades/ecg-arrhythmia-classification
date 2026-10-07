"""
Генерирует итоговые таблицы для курсовой работы.

Запуск (после всех остальных скриптов):
    python generate_report_table.py

Что создаёт:
    reports/table_clean_data.csv      — качество на чистых данных
    reports/table_noise_summary.csv   — устойчивость к шуму (max уровень)
    reports/table_cv_summary.csv      — кросс-валидация
    reports/figures/models/table_clean_data.png
    reports/figures/models/table_noise_summary.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as mtable


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_PATH = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOISE_LABELS = {
    "gaussian":  "Гауссов шум",
    "baseline":  "Дрейф изолинии",
    "powerline": "Сетевая помеха",
    "muscle":    "Мышечный шум",
}


def render_table_as_image(df, title, filename, col_widths=None):
    n_rows, n_cols = df.shape
    fig_height = max(2.5, 0.5 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 2), fig_height))
    ax.axis("off")

    cell_text = df.values.tolist()
    col_labels = list(df.columns)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    # Заголовок — синий фон
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#2E5FA3")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Чередующиеся строки
    for i in range(1, n_rows + 1):
        color = "#EEF2FA" if i % 2 == 0 else "white"
        for j in range(n_cols):
            tbl[(i, j)].set_facecolor(color)

    plt.title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIG_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Изображение таблицы: {out}")


def build_clean_table():
    path = REPORTS_PATH / "clean_data_results.csv"
    if not path.exists():
        print(f"[WARN] Файл не найден: {path}. Сначала запустите evaluate_models.py")
        return None

    df = pd.read_csv(path)
    df["accuracy"]    = df["accuracy"].map("{:.4f}".format)
    df["macro_f1"]    = df["macro_f1"].map("{:.4f}".format)
    df["weighted_f1"] = df["weighted_f1"].map("{:.4f}".format)
    df.columns = ["Модель", "Accuracy", "Macro F1", "Weighted F1"]

    out = REPORTS_PATH / "table_clean_data.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Таблица 1 (чистые данные): {out}")
    print(df.to_string(index=False))
    render_table_as_image(df, "Качество классификации на чистых данных", "table_clean_data.png")
    return df


def build_noise_table():
    path = REPORTS_PATH / "noise_results_all.csv"
    if not path.exists():
        print(f"[WARN] Файл не найден: {path}. Сначала запустите noise_experiments.py")
        return None

    df = pd.read_csv(path)

    rows = []
    for noise_type, noise_label in NOISE_LABELS.items():
        sub = df[df["noise_type"] == noise_type]
        max_level = sub["level"].max()
        sub_max = sub[sub["level"] == max_level]

        for _, row in sub_max.iterrows():
            rows.append({
                "Тип шума":         noise_label,
                "Макс. уровень":    f"{max_level:.2f}",
                "Модель":           row["model"],
                "Accuracy":         f"{row['accuracy']:.4f}",
                "Macro F1":         f"{row['macro_f1']:.4f}",
            })

    result = pd.DataFrame(rows)
    out = REPORTS_PATH / "table_noise_summary.csv"
    result.to_csv(out, index=False)
    print(f"\n[OK] Таблица 2 (шум, макс. уровень): {out}")
    print(result.to_string(index=False))
    render_table_as_image(result, "Устойчивость к шуму (максимальный уровень)", "table_noise_summary.png")
    return result


def build_cv_table():
    path = REPORTS_PATH / "cv_results_summary.csv"
    if not path.exists():
        print(f"[WARN] Файл не найден: {path}. Сначала запустите cross_validation.py")
        return None

    df = pd.read_csv(path)
    df["Accuracy"] = df.apply(
        lambda r: f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}", axis=1
    )
    df["Macro F1"] = df.apply(
        lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}", axis=1
    )
    result = df[["model", "Accuracy", "Macro F1"]].copy()
    result.columns = ["Модель", "Accuracy (mean ± std)", "Macro F1 (mean ± std)"]

    out = REPORTS_PATH / "table_cv_summary.csv"
    result.to_csv(out, index=False)
    print(f"\n[OK] Таблица 3 (кросс-валидация): {out}")
    print(result.to_string(index=False))
    render_table_as_image(result, "5-fold кросс-валидация (mean ± std)", "table_cv_summary.png")
    return result


def main():
    print("=" * 55)
    print("  Генерация итоговых таблиц курсовой работы")
    print("=" * 55)

    build_clean_table()
    build_noise_table()
    build_cv_table()

    print("\n[OK] Все таблицы сохранены в reports/")


if __name__ == "__main__":
    main()
