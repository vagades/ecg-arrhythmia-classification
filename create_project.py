import os

# Путь к проекту
base_path = os.path.dirname(os.path.abspath(__file__))

# Структура проекта
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/utils",
    "models",
    "reports/figures"
]

files = {
    "README.md": "# Курсовая работа\n\nАнализ и классификация биосигналов",
    "requirements.txt": "numpy\npandas\nmatplotlib\nscikit-learn\nwfdb\ntorch",
    "main.py": "# основной файл проекта\n\nif __name__ == '__main__':\n    print('Проект запущен')"
}

# Создание папок
for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)

# Создание файлов
for file_name, content in files.items():
    file_path = os.path.join(base_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Проект успешно создан!")