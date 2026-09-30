# scripts/download_preprocessed.py
#!/usr/bin/env python3
"""
Альтернативный вариант: загрузка предобработанных мел-спектрограмм.
Это позволяет избежать медленного извлечения WAV файлов.
"""

import requests
import zipfile
import io
from pathlib import Path
from tqdm import tqdm

def download_preprocessed_nsynth():
    """Загрузка предобработанного датасета (если доступен)"""
    # Возможные источники предобработанных данных
    urls = [
        "https://example.com/nsynth_melspectrograms_5k.zip",  # Замените на реальный URL
        # Добавьте другие источники по мере необходимости
    ]
    
    output_dir = Path("data/processed/nsynth_preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Поиск предобработанных данных...")
    
    # Проверяем, нет ли уже скачанных данных
    npy_files = list(output_dir.glob("*.npy"))
    if len(npy_files) >= 2:
        print(f"✓ Предобработанные данные уже существуют в {output_dir}")
        return True
    
    # Пока нет реального URL, создаем заглушку
    print("⚠️  Реальные предобработанные данные не доступны.")
    print("   Используйте --fallback для загрузки оригинальных WAV файлов.")
    
    return False

def create_small_dataset_from_existing():
    """
    Создание небольшого датасета из уже существующих 81 семпла
    с сильной аугментацией для увеличения размера.
    """
    print("\nСоздание расширенного датасета из существующих 81 семпла...")
    
    import numpy as np
    import librosa
    from pathlib import Path
    from tqdm import tqdm
    
    # Загружаем существующие данные
    data_dir = Path("data/processed/nsynth_processed")
    specs = np.load(data_dir / "melspectrograms.npy")
    
    print(f"Исходный размер: {specs.shape}")
    
    # Параметры аугментации
    n_augmentations = 60  # Создаем 60 аугментированных версий для каждого семпла
    augmented_specs = []
    
    for i in tqdm(range(len(specs)), desc="Аугментация"):
        spec = specs[i, 0]  # Убираем размерность канала
        
        for j in range(n_augmentations):
            # Различные типы аугментаций
            aug_spec = spec.copy()
            
            # 1. Временные сдвиги (time stretching simulation)
            if j % 3 == 0:
                shift = np.random.randint(-5, 6)
                if shift > 0:
                    aug_spec = np.roll(aug_spec, shift, axis=1)
                    aug_spec[:, :shift] = 0
                elif shift < 0:
                    aug_spec = np.roll(aug_spec, shift, axis=1)
                    aug_spec[:, shift:] = 0
            
            # 2. Частотные маски (frequency masking)
            if j % 4 == 0:
                mask_height = np.random.randint(5, 20)
                mask_start = np.random.randint(0, 128 - mask_height)
                aug_spec[mask_start:mask_start+mask_height, :] = 0
            
            # 3. Временные маски (time masking)
            if j % 5 == 0:
                mask_width = np.random.randint(5, 15)
                mask_start = np.random.randint(0, 63 - mask_width)
                aug_spec[:, mask_start:mask_start+mask_width] = 0
            
            # 4. Добавление шума
            if j % 2 == 0:
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_level, aug_spec.shape)
                aug_spec = aug_spec + noise
                aug_spec = np.clip(aug_spec, 0, 1)
            
            # 5. Изменение контраста
            if j % 3 == 1:
                contrast = np.random.uniform(0.8, 1.2)
                aug_spec = aug_spec * contrast
                aug_spec = np.clip(aug_spec, 0, 1)
            
            augmented_specs.append(aug_spec[np.newaxis, np.newaxis, :, :])
    
    # Объединяем оригинальные и аугментированные данные
    all_specs = np.concatenate([specs] + augmented_specs, axis=0)
    
    print(f"Размер после аугментации: {all_specs.shape}")
    print(f"Увеличение в {len(all_specs) / len(specs):.1f} раз")
    
    # Сохраняем
    output_dir = Path("data/processed/nsynth_augmented")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "melspectrograms.npy", all_specs)
    
    # Копируем метаданные и умножаем их
    import shutil
    import json
    import pandas as pd
    
    # Метаданные
    with open(data_dir / "processed_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Создаем аугментированные метаданные
    augmented_metadata = []
    for i, meta in enumerate(metadata):
        for j in range(n_augmentations):
            new_meta = meta.copy()
            new_meta['original_filename'] = f"aug_{j}_{meta['original_filename']}"
            new_meta['augmentation_id'] = j
            augmented_metadata.append(new_meta)
    
    # Объединяем
    all_metadata = metadata + augmented_metadata
    
    with open(output_dir / "processed_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"✓ Создан аугментированный датасет: {output_dir}")
    print(f"   Всего семплов: {len(all_specs)}")
    
    return True

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Альтернативные методы получения данных')
    parser.add_argument('--method', choices=['preprocessed', 'augment', 'download'],
                       default='augment', help='Метод получения данных')
    parser.add_argument('--augmentations', type=int, default=60,
                       help='Количество аугментаций на семпл (только для метода augment)')
    
    args = parser.parse_args()
    
    if args.method == 'preprocessed':
        success = download_preprocessed_nsynth()
    elif args.method == 'augment':
        success = create_small_dataset_from_existing()
    else:
        print("Используйте скрипт download_nsynth_train_fast.py для загрузки")
        success = False
    
    if success:
        print("\n✓ Данные успешно подготовлены")
        print("Можно начинать обучение модели.")
    else:
        print("\n✗ Не удалось подготовить данные")

if __name__ == "__main__":
    main()