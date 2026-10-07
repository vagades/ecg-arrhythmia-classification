"""
Вейвлет-анализ ЭКГ-ударов: визуализация для курсовой работы.

Генерирует три типа рисунков:
    1. wavelet_decomposition_{class}.png — 4-уровневое разложение db4
       для типичного удара каждого класса.
    2. wavelet_avg_energy.png — средняя энергия коэффициентов по классам
       (показывает, в каких подполосах классы различаются).
    3. wavelet_cwt_gallery.png — непрерывное вейвлет-преобразование (CWT)
       для 5 классов (скаллограммы).

Теоретическая основа:
    Захарова Т.В., Шестаков О.В. «Теория вейвлетов и её применение в обработке
    сигналов». — М.: МастерПринт, 2018.

Запуск:
    python plot_wavelet_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_PATH = PROJECT_ROOT / "data" / "processed" / "ecg" / "beats.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FS = 360
WAVELET_DWT = "db4"
LEVEL = 4
WAVELET_CWT = "morl"

CLASS_COLORS = {
    "N": "#4C9BE8",
    "A": "#F4A460",
    "V": "#E8604C",
    "L": "#6CBF6C",
    "R": "#B06CBF",
}
CLASS_LABELS = {
    "N": "Норма (N)",
    "A": "Предсердная экстрасистола (A)",
    "V": "Желудочковая экстрасистола (V)",
    "L": "Блокада левой ножки (L)",
    "R": "Блокада правой ножки (R)",
}
SUBBAND_NAMES = ["cD1\n(90–180 Гц)", "cD2\n(45–90 Гц)",
                 "cD3\n(22.5–45 Гц)", "cD4\n(11–22.5 Гц)", "cA4\n(0–11 Гц)"]


def load_data():
    df = pd.read_csv(BEATS_PATH)
    signal_cols = [c for c in df.columns if c.startswith("x_")]
    X = df[signal_cols].values.astype(np.float32)
    labels = df["label"].values
    return X, labels


def get_median_beat(X, labels, cls):
    """Возвращает медианный удар класса cls."""
    idx = np.where(labels == cls)[0]
    beats = X[idx]
    return np.median(beats, axis=0)


def plot_dwt_decomposition(beat, cls):
    """Рисунок: исходный сигнал + все уровни разложения."""
    coeffs = pywt.wavedec(beat, WAVELET_DWT, level=LEVEL)
    cA4, cD4, cD3, cD2, cD1 = coeffs

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(
        f"Вейвлет-разложение ЭКГ-удара (db4, 4 уровня)\nКласс: {CLASS_LABELS[cls]}",
        fontsize=12,
    )

    subplots = [
        (beat, f"Исходный сигнал ({len(beat)} отсчётов)", CLASS_COLORS[cls]),
        (cD1, "cD1  (90–180 Гц) — шум", "#999999"),
        (cD2, "cD2  (45–90 Гц) — QRS высокочастотные", "#E8604C"),
        (cD3, "cD3  (22.5–45 Гц) — комплекс QRS", "#F4A460"),
        (cD4, "cD4  (11.25–22.5 Гц) — зубцы P и T", "#6CBF6C"),
        (cA4, "cA4  (0–11.25 Гц) — изолиния", "#4C9BE8"),
    ]

    for ax, (data, label, color) in zip(axes, subplots):
        ax.plot(data, color=color, linewidth=0.9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = FIG_DIR / f"wavelet_decomposition_{cls}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def plot_energy_by_class(X, labels, classes):
    """
    Средняя энергия коэффициентов в каждой подполосе по классам.
    Энергия = среднее квадрата коэффициентов.
    """
    energy = {cls: [] for cls in classes}

    for cls in classes:
        idx = np.where(labels == cls)[0]
        # Ограничим выборку для скорости
        sample_idx = idx[:2000] if len(idx) > 2000 else idx
        beats = X[sample_idx]
        band_energies = []
        for band_pos in range(LEVEL + 1):
            vals = []
            for beat in beats:
                c = pywt.wavedec(beat, WAVELET_DWT, level=LEVEL)
                # c = [cA4, cD4, cD3, cD2, cD1]
                vals.append(np.mean(c[band_pos] ** 2))
            band_energies.append(np.mean(vals))
        energy[cls] = band_energies

    # Порядок подполос: cA4, cD4, cD3, cD2, cD1
    band_labels = ["cA4\n(0–11 Гц)", "cD4\n(11–22.5 Гц)",
                   "cD3\n(22.5–45 Гц)", "cD2\n(45–90 Гц)", "cD1\n(90–180 Гц)"]

    x = np.arange(LEVEL + 1)
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, cls in enumerate(classes):
        offset = (i - len(classes) / 2 + 0.5) * width
        ax.bar(x + offset, energy[cls], width,
               label=CLASS_LABELS[cls], color=CLASS_COLORS[cls], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel("Средняя энергия коэффициентов")
    ax.set_title("Распределение энергии ДВП по частотным подполосам и классам ЭКГ\n"
                 "(вейвлет db4, 4 уровня разложения, fs=360 Гц)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = FIG_DIR / "wavelet_avg_energy.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def plot_cwt_gallery(X, labels, classes):
    """
    Скаллограммы (CWT) для медианного удара каждого класса.
    """
    scales = np.arange(1, 64)
    t = np.arange(240) / FS * 1000  # мс

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        beat = get_median_beat(X, labels, cls)

        # CWT
        coef, freqs = pywt.cwt(beat, scales, WAVELET_CWT, sampling_period=1 / FS)

        ax_sig = axes[i]
        ax_cwt = axes[i + len(classes)]

        # Верхняя строка: сигнал
        ax_sig.plot(t, beat, color=CLASS_COLORS[cls], linewidth=1.2)
        ax_sig.set_title(CLASS_LABELS[cls], fontsize=9)
        ax_sig.set_xlabel("мс", fontsize=7)
        ax_sig.set_ylabel("мВ", fontsize=7)
        ax_sig.tick_params(labelsize=6)
        ax_sig.grid(True, alpha=0.3)

        # Нижняя строка: скаллограмма
        ax_cwt.contourf(t, freqs, np.abs(coef), levels=30, cmap="jet")
        ax_cwt.set_xlabel("мс", fontsize=7)
        ax_cwt.set_ylabel("Гц", fontsize=7)
        ax_cwt.set_ylim(0, 60)
        ax_cwt.tick_params(labelsize=6)

    fig.suptitle(
        "Скаллограммы ЭКГ-ударов (CWT, вейвлет Морле)\n"
        "Верхний ряд — сигнал, нижний — частотно-временное распределение энергии",
        fontsize=11,
    )
    plt.tight_layout()
    path = FIG_DIR / "wavelet_cwt_gallery.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {path}")


def main():
    print("[INFO] Загружаю данные...")
    X, labels = load_data()
    print(f"[INFO] X: {X.shape}")

    classes = ["N", "A", "V", "L", "R"]

    # 1. Разложение db4 для медианного удара каждого класса
    print("[INFO] Строю вейвлет-разложения по классам...")
    for cls in classes:
        beat = get_median_beat(X, labels, cls)
        plot_dwt_decomposition(beat, cls)

    # 2. Энергия по подполосам и классам
    print("[INFO] Строю диаграмму энергии подполос...")
    plot_energy_by_class(X, labels, classes)

    # 3. Скаллограммы CWT
    print("[INFO] Строю скаллограммы CWT...")
    plot_cwt_gallery(X, labels, classes)

    print("\n[OK] Все вейвлет-рисунки сохранены в:", FIG_DIR)


if __name__ == "__main__":
    main()
