# scripts/download_nsynth_subset.py
#!/usr/bin/env python3
"""
Скрипт для загрузки и подготовки подмножества датасета NSynth.
Загружает только выбранные инструментальные семейства и ноты.
"""

import os
import sys
import yaml
import json
import tarfile
import urllib.request
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class NSynthDownloader:
    def __init__(self, config_path="config/nsynth_config.yaml"):
        """Инициализация с конфигурационным файлом"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['nsynth']
        
        # Создаем директории
        self.archive_path = Path(self.config['archive_path'])
        self.extract_dir = Path(self.config['extract_dir'])
        self.output_dir = Path(self.config['output_dir'])
        
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки выборки
        self.desired_families = set(self.config['desired_families'])
        self.desired_notes = set(self.config['desired_notes'])
        self.samples_per_family_note = self.config['samples_per_family_note']
        
        random.seed(self.config['random_seed'])
        
    def download_archive(self, force_redownload=False):
        """Скачивание архива NSynth, если его нет"""
        if self.archive_path.exists() and not force_redownload:
            print(f"Архив уже существует: {self.archive_path}")
            archive_size = self.archive_path.stat().st_size / (1024**3)
            print(f"Размер: {archive_size:.2f} GB")
            return True
            
        print(f"Скачивание архива NSynth...")
        print(f"URL: {self.config['dataset_url']}")
        print(f"Сохраняем в: {self.archive_path}")
        
        try:
            # Создаем прогресс-бар для загрузки
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
            
            with DownloadProgressBar(unit='B', unit_scale=True, 
                                    miniters=1, desc='Скачивание') as t:
                urllib.request.urlretrieve(
                    self.config['dataset_url'],
                    filename=self.archive_path,
                    reporthook=t.update_to
                )
            
            print("Скачивание завершено!")
            return True
            
        except Exception as e:
            print(f"Ошибка при скачивании: {e}")
            return False
    
    def extract_metadata(self):
        """Извлечение метаданных из архива (без распаковки всех файлов)"""
        print("Извлечение метаданных...")
        
        try:
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                # Ищем файл с метаданными
                metadata_members = [m for m in tar.getmembers() 
                                  if 'examples.json' in m.name]
                
                if not metadata_members:
                    print("Файл метаданных не найден в архиве!")
                    return None
                
                # Извлекаем только метаданные
                metadata_file = metadata_members[0]
                f = tar.extractfile(metadata_file)
                metadata = json.load(f)
                print(f"Загружено {len(metadata)} записей метаданных")
                return metadata
                
        except Exception as e:
            print(f"Ошибка при извлечении метаданных: {e}")
            return None
    
    def filter_samples(self, metadata):
        """Фильтрация семплов по выбранным семействам и нотам"""
        print("Фильтрация семплов...")
        
        filtered_samples = []
        
        for sample_id, sample_info in tqdm(metadata.items(), desc="Фильтрация"):
            # Проверяем условия
            instrument_family = sample_info.get('instrument_family_str', 
                                              sample_info.get('instrument_family', ''))
            pitch = sample_info.get('pitch', -1)
            
            if (instrument_family in self.desired_families and 
                pitch in self.desired_notes):
                
                # Добавляем дополнительную информацию
                sample_data = {
                    'id': sample_id,
                    'instrument_family': instrument_family,
                    'pitch': pitch,
                    'instrument_source': sample_info.get('instrument_source_str', 
                                                       sample_info.get('instrument_source', '')),
                    'velocity': sample_info.get('velocity', 64),
                    'qualities': sample_info.get('qualities_str', 
                                               sample_info.get('qualities', [])),
                    'instrument': sample_info.get('instrument_str', 
                                                sample_info.get('instrument', -1))
                }
                filtered_samples.append(sample_data)
        
        print(f"Найдено {len(filtered_samples)} подходящих семплов")
        return filtered_samples
    
    def select_samples(self, filtered_samples):
        """Выбор конкретных семплов для загрузки"""
        print("Выбор семплов...")
        
        # Группируем по семейству и ноте
        grouped = {}
        for sample in filtered_samples:
            key = (sample['instrument_family'], sample['pitch'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(sample)
        
        # Выбираем заданное количество семплов из каждой группы
        selected_samples = []
        for (family, pitch), samples in grouped.items():
            # Перемешиваем для случайности (но с фиксированным seed)
            random.shuffle(samples)
            selected = samples[:self.samples_per_family_note]
            selected_samples.extend(selected)
            
            print(f"  {family} (нота {pitch}): {len(selected)} семплов")
        
        # Ограничиваем общее количество
        if len(selected_samples) > self.config['max_total_samples']:
            selected_samples = selected_samples[:self.config['max_total_samples']]
        
        print(f"Итого выбрано {len(selected_samples)} семплов")
        return selected_samples
    
    def extract_selected_samples(self, selected_samples):
        """Извлечение выбранных аудиофайлов из архива"""
        print("Извлечение аудиофайлов...")
        
        # Создаем структуру директорий по семействам инструментов
        for family in self.desired_families:
            family_dir = self.output_dir / family
            family_dir.mkdir(exist_ok=True)
        
        extracted_count = 0
        
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for sample in tqdm(selected_samples, desc="Извлечение файлов"):
                sample_id = sample['id']
                family = sample['instrument_family']
                pitch = sample['pitch']
                
                # Путь к файлу в архиве
                audio_path_in_archive = f"nsynth-test/audio/{sample_id}.wav"
                
                try:
                    # Извлекаем файл
                    tar.extract(audio_path_in_archive, path=self.extract_dir)
                    
                    # Перемещаем в целевую директорию
                    src_path = self.extract_dir / audio_path_in_archive
                    dst_path = self.output_dir / family / f"{sample_id}_pitch{pitch}.wav"
                    
                    shutil.move(str(src_path), str(dst_path))
                    extracted_count += 1
                    
                except Exception as e:
                    print(f"Ошибка при извлечении {sample_id}: {e}")
        
        # Очищаем временные файлы
        if (self.extract_dir / "nsynth-test").exists():
            shutil.rmtree(self.extract_dir / "nsynth-test")
        
        print(f"Извлечено {extracted_count} аудиофайлов")
        return extracted_count
    
    def create_metadata_csv(self, selected_samples):
        """Создание CSV файла с метаданными выбранных семплов"""
        print("Создание файла метаданных...")
        
        import pandas as pd
        
        metadata_list = []
        for sample in selected_samples:
            # Добавляем путь к файлу
            family = sample['instrument_family']
            sample_id = sample['id']
            pitch = sample['pitch']
            
            file_path = f"{family}/{sample_id}_pitch{pitch}.wav"
            
            metadata = {
                'file_path': file_path,
                'instrument_family': family,
                'instrument_source': sample['instrument_source'],
                'pitch': pitch,
                'velocity': sample['velocity'],
                'qualities': ','.join(map(str, sample['qualities'])),
                'instrument_id': sample['instrument']
            }
            metadata_list.append(metadata)
        
        df = pd.DataFrame(metadata_list)
        csv_path = self.output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Метаданные сохранены в: {csv_path}")
        print(f"Колонки: {list(df.columns)}")
        
        return df
    
    def create_attribute_mapping(self):
        """Создание маппинга для атрибутов инструментов"""
        # Простой маппинг для начала
        attribute_map = {
            'instrument_family': {
                'bass': 'plucked',
                'brass': 'brass',
                'keyboard': 'keyboard',
                'mallet': 'percussion',
                'synth_lead': 'synth',
                'string': 'bowed',
                'vocal': 'vocal'
            },
            'instrument_source': {
                'acoustic': 'acoustic',
                'electronic': 'electronic',
                'synthetic': 'synthetic'
            }
        }
        
        mapping_path = self.output_dir / "attribute_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(attribute_map, f, indent=2)
        
        print(f"Маппинг атрибутов сохранен в: {mapping_path}")
        return attribute_map
    
    def run(self, force_redownload=False):
        """Основной метод запуска пайплайна"""
        print("=" * 60)
        print("ПОДГОТОВКА ПОДМНОЖЕСТВА NSYNTH")
        print("=" * 60)
        
        # Шаг 1: Скачивание архива
        if not self.download_archive(force_redownload):
            return False
        
        # Шаг 2: Извлечение метаданных
        metadata = self.extract_metadata()
        if metadata is None:
            return False
        
        # Шаг 3: Фильтрация семплов
        filtered_samples = self.filter_samples(metadata)
        if not filtered_samples:
            print("Нет семплов, соответствующих критериям!")
            return False
        
        # Шаг 4: Выбор конкретных семплов
        selected_samples = self.select_samples(filtered_samples)
        
        # Шаг 5: Извлечение аудиофайлов
        extracted_count = self.extract_selected_samples(selected_samples)
        if extracted_count == 0:
            return False
        
        # Шаг 6: Создание метаданных
        self.create_metadata_csv(selected_samples)
        
        # Шаг 7: Создание маппинга атрибутов
        self.create_attribute_mapping()
        
        print("\n" + "=" * 60)
        print("ГОТОВО!")
        print("=" * 60)
        print(f"Всего семплов: {extracted_count}")
        print(f"Директория с данными: {self.output_dir}")
        print("\nСтруктура данных:")
        
        # Выводим структуру директорий
        for family in self.desired_families:
            family_dir = self.output_dir / family
            if family_dir.exists():
                wav_files = list(family_dir.glob("*.wav"))
                if wav_files:
                    print(f"  {family}: {len(wav_files)} файлов")
        
        print(f"\nСледующие шаги:")
        print(f"1. Проверить файлы в {self.output_dir}")
        print(f"2. Запустить DataBuilder для обработки")
        print(f"3. Начать обучение модели")
        
        return True

def main():
    """Точка входа"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка подмножества NSynth')
    parser.add_argument('--config', default='config/nsynth_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--force', action='store_true',
                       help='Перезаписать существующий архив')
    
    args = parser.parse_args()
    
    # Проверяем наличие конфигурационного файла
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Конфигурационный файл не найден: {config_path}")
        print("Создаю базовый конфиг...")
        
        # Создаем директорию для конфига
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Копируем базовый конфиг
        base_config = """nsynth:
  dataset_url: "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
  archive_path: "data/raw/nsynth-test.tar.gz"
  extract_dir: "data/raw/nsynth-test"
  output_dir: "data/processed/nsynth_subset"
  desired_families:
    - "bass"
    - "brass"
    - "keyboard"
    - "mallet"
    - "synth_lead"
    - "string"
    - "vocal"
  desired_notes: [60, 64, 67]
  samples_per_family_note: 1
  random_seed: 42
  max_total_samples: 100
"""
        with open(config_path, 'w') as f:
            f.write(base_config)
        
        print(f"Создан конфиг: {config_path}")
        print("Пожалуйста, проверьте настройки и запустите скрипт снова.")
        return
    
    # Запускаем загрузчик
    downloader = NSynthDownloader(config_path)
    success = downloader.run(force_redownload=args.force)
    
    if not success:
        print("Произошла ошибка при подготовке данных.")
        sys.exit(1)

if __name__ == "__main__":
    main()