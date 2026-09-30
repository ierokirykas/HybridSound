# scripts/run_training.py
#!/usr/bin/env python3
"""Скрипт для запуска обучения"""

import subprocess
import sys
from pathlib import Path

def main():
    # Пути к данным (после пайплайна)
    spectrograms_path = "data/processed/nsynth_processed/melspectrograms.npy"
    metadata_path = "data/processed/nsynth_processed/processed_metadata.json"
    
    # Проверка наличия данных
    if not Path(spectrograms_path).exists():
        print(f"Ошибка: файл спектрограмм не найден: {spectrograms_path}")
        print("Сначала запустите пайплайн обработки данных.")
        sys.exit(1)
    
    if not Path(metadata_path).exists():
        print(f"Ошибка: файл метаданных не найден: {metadata_path}")
        sys.exit(1)
    
    # Параметры обучения (можно менять)
    cmd = [
        sys.executable, "scripts/train_vae.py",
        "--spectrograms", spectrograms_path,
        "--metadata", metadata_path,
        "--latent-dim", "32",           # Маленькая размерность для 81 семпла
        "--beta", "4.0",               # Сильная регуляризация
        "--epochs", "300",
        "--batch-size", "8",
        "--lr", "1e-4",
        "--patience", "40",
        "--save-dir", "experiments/vae_training"
    ]
    
    print("Запуск обучения VAE...")
    print("Команда:", " ".join(cmd))
    print("\n" + "="*60)
    
    # Запуск
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("="*60)
        
        # Показываем где сохранены результаты
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Результаты сохранены в: experiments/vae_training/vae_{timestamp}")
    else:
        print("\nОбучение завершилось с ошибкой.")

if __name__ == "__main__":
    main()