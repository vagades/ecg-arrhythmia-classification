from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ecg" / "mit_bih_arrhythmia"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ecg"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Оставим самые полезные для курсовой классы
ALLOWED_SYMBOLS = {
    "N",  # normal beat
    "L",  # left bundle branch block beat
    "R",  # right bundle branch block beat
    "A",  # atrial premature beat
    "V",  # premature ventricular contraction
}


def get_record_names():
    """
    Берёт список записей по .hea файлам.
    """
    return sorted({p.stem for p in RAW_DIR.glob("*.hea")})


def extract_beats_from_record(
    record_name: str,
    channel: int = 0,
    left_size: int = 100,
    right_size: int = 140,
):
    """
    Для одной записи:
    - читает сигнал и аннотации
    - вырезает окно вокруг каждого удара
    - возвращает список словарей
    """
    record_path = str(RAW_DIR / record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, channel]
    beats = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in ALLOWED_SYMBOLS:
            continue

        start = sample - left_size
        end = sample + right_size

        # Пропускаем удары, если окно вылезает за границы сигнала
        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        row = {
            "record_name": record_name,
            "sample": sample,
            "label": symbol,
        }

        for i, value in enumerate(beat):
            row[f"x_{i}"] = float(value)

        beats.append(row)

    return beats


def main():
    record_names = get_record_names()

    if not record_names:
        print("[ERROR] Не найдены записи в папке:")
        print(RAW_DIR)
        return

    print(f"[INFO] Найдено записей: {len(record_names)}")
    all_beats = []

    for record_name in record_names:
        print(f"[INFO] Обрабатывается запись {record_name}...")
        beats = extract_beats_from_record(record_name)
        print(f"[OK] Из записи {record_name} извлечено {len(beats)} ударов")
        all_beats.extend(beats)

    if not all_beats:
        print("[ERROR] Не удалось извлечь ни одного удара")
        return

    df = pd.DataFrame(all_beats)

    output_path = PROCESSED_DIR / "beats.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n[OK] Готово!")
    print(f"[OK] Файл сохранён: {output_path}")
    print(f"[INFO] Размер таблицы: {df.shape}")

    class_counts = Counter(df["label"])
    print("[INFO] Распределение классов:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()