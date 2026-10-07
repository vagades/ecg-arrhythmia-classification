from pathlib import Path
import os
import sys
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ECG_DIR = RAW_DIR / "ecg" / "mit_bih_arrhythmia"
EEG_DIR = RAW_DIR / "eeg" / "eeg_motor_movement_imagery"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def install_if_missing(package_name: str) -> None:
    try:
        __import__(package_name)
    except ImportError:
        print(f"[INFO] Пакет {package_name} не найден. Устанавливаю...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def download_mit_bih() -> None:
    """
    Скачивает MIT-BIH Arrhythmia Database через wfdb.
    Файлы сохраняются в:
    data/raw/ecg/mit_bih_arrhythmia
    """
    install_if_missing("wfdb")
    import wfdb

    ensure_dir(ECG_DIR)

    print("[INFO] Скачиваю MIT-BIH Arrhythmia Database...")
    wfdb.dl_database(
        "mitdb",
        dl_dir=str(ECG_DIR),
    )
    print(f"[OK] MIT-BIH сохранён в: {ECG_DIR}")


def download_eeg_motor_imagery() -> None:
    """
    Скачивает EEG Motor Movement/Imagery Dataset через mne.
    Данные сначала скачиваются во внутреннюю папку MNE,
    поэтому мы задаём свою папку через переменную окружения.
    """
    install_if_missing("mne")
    import mne
    from mne.datasets import eegbci

    ensure_dir(EEG_DIR)

    # Говорим MNE хранить датасет именно в нашей папке
    os.environ["MNE_DATA"] = str(EEG_DIR)

    print("[INFO] Скачиваю EEG Motor Movement/Imagery Dataset...")
    # Несколько субъектов для старта, чтобы не качать всё сразу
    # Потом можно расширить список subjects
    subjects = [1, 2, 3]
    runs = [1, 2, 3, 4, 5, 6]

    for subject in subjects:
        eegbci.load_data(subjects=[subject], runs=runs, path=str(EEG_DIR))
        print(f"[OK] Субъект {subject} скачан")

    print(f"[OK] EEG-датасет сохранён в: {EEG_DIR}")


def create_extra_project_files() -> None:
    """
    Создаёт полезные папки и файлы-заглушки для дальнейшей обработки.
    """
    extra_dirs = [
        PROJECT_ROOT / "data" / "processed" / "ecg",
        PROJECT_ROOT / "data" / "processed" / "eeg",
        PROJECT_ROOT / "src" / "data_loading",
        PROJECT_ROOT / "src" / "preprocessing",
        PROJECT_ROOT / "src" / "training",
    ]

    for d in extra_dirs:
        ensure_dir(d)

    files = {
        PROJECT_ROOT / "src" / "data_loading" / "__init__.py": "",
        PROJECT_ROOT / "src" / "preprocessing" / "__init__.py": "",
        PROJECT_ROOT / "src" / "training" / "__init__.py": "",
        PROJECT_ROOT / "src" / "data_loading" / "load_ecg.py":
            "# Тут потом будет код чтения MIT-BIH\n",
        PROJECT_ROOT / "src" / "data_loading" / "load_eeg.py":
            "# Тут потом будет код чтения EEG\n",
    }

    for file_path, content in files.items():
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")

    print("[OK] Дополнительные папки и файлы созданы")


def main() -> None:
    ensure_dir(RAW_DIR)
    create_extra_project_files()

    print("\nВыбери, что скачать:")
    print("1 - Только ECG (MIT-BIH)")
    print("2 - Только EEG")
    print("3 - ECG + EEG")
    choice = input("Введи 1 / 2 / 3: ").strip()

    if choice == "1":
        download_mit_bih()
    elif choice == "2":
        download_eeg_motor_imagery()
    elif choice == "3":
        download_mit_bih()
        download_eeg_motor_imagery()
    else:
        print("[ERROR] Неверный выбор. Нужно ввести 1, 2 или 3.")


if __name__ == "__main__":
    main()