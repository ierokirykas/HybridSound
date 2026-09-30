# scripts/run_pipeline.py
#!/usr/bin/env python3
"""Скрипт для запуска полного пайплайна"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_pipeline import DataPipeline

def run_nsynth_pipeline():
    """Запуск пайплайна для данных NSynth"""
    print("Запуск пайплайна обработки NSynth данных...")
    
    # Пути к данным

    # input_dir = "data/processed/nsynth_subset"
    # metadata_csv = "data/processed/nsynth_subset/metadata.csv"
    # output_dir = "data/processed/nsynth_processed"

    input_dir = "data/processed/nsynth_train_25k"
    metadata_csv = "data/processed/nsynth_train_25k/metadata.csv"
    output_dir = "data/processed/nsynth_ready_25k"
    # Конфигурационный файл
    config_path = "config/data_config.yaml"
    
    # Создание и запуск пайплайна
    pipeline = DataPipeline(config_path)
    specs, metadata = pipeline.process_dataset(input_dir, metadata_csv, output_dir)
    
    return specs, metadata

def test_small_batch():
    """Тест на маленьком подмножестве"""
    print("Тестирование на маленьком подмножестве...")
    
    # Создаем тестовый конфиг
    test_config = {
        'data_pipeline': {
            'sr': 16000,
            'target_sr': 44100,
            'resample': False,
            'duration': 1.0,  # Короткая длительность для теста
            'n_mels': 64,     # Уменьшенное количество мел-банков
            'hop_length': 256,
            'n_fft': 1024,
            'fmax': 8000,
            'mono': True,
            'normalize': True,
            'target_rms': 0.05,
            'min_duration': 0.5,
            'max_clipping': 0.9,
            'attributes': [
                {
                    'name': 'instrument_family',
                    'source_column': 'instrument_family',
                    'values': ['bass', 'string']
                }
            ]
        }
    }
    
    # Сохраняем временный конфиг
    import yaml
    test_config_path = "config/test_pipeline_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        # Тестовые данные (первые 5 файлов)
        import pandas as pd
        metadata_df = pd.read_csv("data/processed/nsynth_subset/metadata.csv")
        test_metadata = metadata_df.head(5).copy()
        test_metadata_path = "data/test_metadata.csv"
        test_metadata.to_csv(test_metadata_path, index=False)
        
        # Запускаем пайплайн
        pipeline = DataPipeline(test_config_path)
        specs, metadata = pipeline.process_dataset(
            input_dir="data/processed/nsynth_subset",
            metadata_csv=test_metadata_path,
            output_dir="data/test_processed"
        )
        
        if specs is not None:
            print(f"✓ Тест успешен! Обработано {len(metadata)} файлов")
            print(f"  Размерность спектрограмм: {specs.shape}")
        else:
            print("✗ Тест не удался")
            
    except Exception as e:
        print(f"✗ Ошибка при тестировании: {e}")
    finally:
        # Очистка
        import shutil
        Path(test_config_path).unlink(missing_ok=True)
        Path(test_metadata_path).unlink(missing_ok=True)
        if Path("data/test_processed").exists():
            shutil.rmtree("data/test_processed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Запуск пайплайна обработки данных')
    parser.add_argument('--test', action='store_true', help='Запустить тестовый режим')
    parser.add_argument('--full', action='store_true', help='Запустить полную обработку')
    
    args = parser.parse_args()
    
    if args.test:
        test_small_batch()
    elif args.full:
        run_nsynth_pipeline()
    else:
        print("Укажите режим: --test или --full")