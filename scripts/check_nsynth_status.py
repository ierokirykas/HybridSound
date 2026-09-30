# scripts/check_nsynth_status.py
#!/usr/bin/env python3
"""Скрипт для проверки статуса загрузки NSynth"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.download_nsynth_train import NSynthTrainDownloader

def check_download_progress():
    """Проверка прогресса загрузки"""
    print("ПРОВЕРКА СТАТУСА ЗАГРУЗКИ NSYNTH")
    print("=" * 60)
    
    # Проверяем конфигурации
    configs = [
        ("Test split (малый)", "config/nsynth_config.yaml"),
        ("Train split (большой)", "config/nsynth_train_config.yaml")
    ]
    
    for name, config_path in configs:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"\n{name}: конфиг не найден")
            continue
        
        print(f"\n{name}:")
        print(f"  Конфиг: {config_path}")
        
        try:
            downloader = NSynthTrainDownloader(config_path)
            
            # Проверяем архив
            if downloader.archive_path.exists():
                size_gb = downloader.archive_path.stat().st_size / (1024**3)
                print(f"  Архив: ✓ ({size_gb:.1f} GB)")
            else:
                print(f"  Архив: ✗ (не скачан)")
            
            # Проверяем выходную директорию
            if downloader.output_dir.exists():
                wav_files = list(downloader.output_dir.rglob("*.wav"))
                metadata_files = list(downloader.output_dir.glob("*.csv"))
                
                print(f"  Датасет: ✓ ({len(wav_files)} файлов)")
                if metadata_files:
                    print(f"  Метаданные: ✓ ({len(metadata_files)} файлов)")
            else:
                print(f"  Датасет: ✗ (не создан)")
                
        except Exception as e:
            print(f"  Ошибка: {e}")
    
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("1. Для загрузки test split (81 семпл):")
    print("   python scripts/download_nsynth_subset.py")
    print("\n2. Для загрузки train split (5000 семплов):")
    print("   python scripts/download_nsynth_train.py")
    print("\n3. Для проверки без загрузки:")
    print("   python scripts/download_nsynth_train.py --check-only")

if __name__ == "__main__":
    check_download_progress()