from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "ecg_preprocessing"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_beats():
    if not BEATS_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {BEATS_PATH}")
    return pd.read_csv(BEATS_PATH)


def get_signal_columns(df: pd.DataFrame):
    return [col for col in df.columns if col.startswith("x_")]


def plot_examples_by_class(
    df: pd.DataFrame,
    classes=("N", "A", "V", "L", "R"),
    n_examples=5,
    save=True,
):
    signal_cols = get_signal_columns(df)

    available_classes = [c for c in classes if c in df["label"].unique()]
    n_rows = len(available_classes)

    if n_rows == 0:
        print("[ERROR] Нет нужных классов в таблице.")
        return

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 3 * n_rows), squeeze=False
    )

    for i, cls in enumerate(available_classes):
        ax = axes[i, 0]
        class_df = df[df["label"] == cls].head(n_examples)

        for _, row in class_df.iterrows():
            beat = row[signal_cols].values.astype(float)
            ax.plot(beat, alpha=0.8)

        ax.set_title(f"Класс {cls}: {len(class_df)} примеров")
        ax.set_xlabel("Отсчёт внутри окна")
        ax.set_ylabel("Амплитуда")
        ax.grid(True)

    plt.tight_layout()

    if save:
        output_path = FIG_DIR / "beats_by_class_examples.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Картинка сохранена: {output_path}")

    plt.close()


def plot_mean_beat_by_class(
    df: pd.DataFrame,
    classes=("N", "A", "V", "L", "R"),
    save=True,
):
    signal_cols = get_signal_columns(df)

    plt.figure(figsize=(12, 6))

    for cls in classes:
        class_df = df[df["label"] == cls]
        if len(class_df) == 0:
            continue

        mean_beat = class_df[signal_cols].mean(axis=0).values.astype(float)
        plt.plot(mean_beat, label=f"{cls} (n={len(class_df)})")

    plt.title("Средний сердечный удар по классам")
    plt.xlabel("Отсчёт внутри окна")
    plt.ylabel("Средняя амплитуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        output_path = FIG_DIR / "mean_beats_by_class.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Картинка сохранена: {output_path}")

    plt.close()


def main():
    df = load_beats()

    print("[INFO] Размер таблицы:", df.shape)
    print("[INFO] Классы:", sorted(df["label"].unique()))

    print("\n[INFO] Количество объектов по классам:")
    print(df["label"].value_counts())

    plot_examples_by_class(df, n_examples=5, save=True)
    plot_mean_beat_by_class(df, save=True)


if __name__ == "__main__":
    main()  