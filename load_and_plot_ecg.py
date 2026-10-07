from pathlib import Path

import matplotlib.pyplot as plt
import wfdb
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "ecg_raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ecg" / "mit_bih_arrhythmia"


def load_record(record_name: str = "100"):
    """
    Загружает запись ЭКГ и аннотации для выбранного record.
    """
    record_path = str(DATA_DIR / record_name)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    return record, annotation


def print_record_info(record, annotation, record_name: str):
    """
    Печатает основную информацию о записи.
    """
    print(f"Запись: {record_name}")
    print(f"Частота дискретизации: {record.fs} Гц")
    print(f"Количество каналов: {record.n_sig}")
    print(f"Названия каналов: {record.sig_name}")
    print(f"Длина сигнала: {record.sig_len} отсчётов")
    print(f"Длительность: {record.sig_len / record.fs:.2f} сек")
    print(f"Количество аннотаций: {len(annotation.sample)}")
    print("Первые 20 меток:")
    print(annotation.symbol[:20])


def plot_ecg_segment(record, annotation, start_sec: float = 0, duration_sec: float = 10):
    """
    Строит график фрагмента ЭКГ и отмечает аннотации.
    """
    fs = record.fs
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)

    signal = record.p_signal[start_sample:end_sample, 0]
    time_axis = [i / fs for i in range(start_sample, end_sample)]

    ann_samples = []
    ann_symbols = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if start_sample <= sample < end_sample:
            ann_samples.append(sample)
            ann_symbols.append(symbol)

    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, signal, label="ECG, канал 0")

    for sample, symbol in zip(ann_samples, ann_symbols):
        x = sample / fs
        y = record.p_signal[sample, 0]
        plt.scatter(x, y, marker="o")
        plt.text(x, y + 0.15, symbol, fontsize=8)

    plt.title(f"Фрагмент ЭКГ: {start_sec}–{start_sec + duration_sec} сек")
    plt.xlabel("Время, сек")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    file_path = FIG_DIR / "record_100_first_10_sec.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"[OK] График сохранён: {file_path}")
    plt.show()


def main():
    record_name = "100"

    record, annotation = load_record(record_name)
    print_record_info(record, annotation, record_name)
    plot_ecg_segment(record, annotation, start_sec=0, duration_sec=10)


if __name__ == "__main__":
    main()