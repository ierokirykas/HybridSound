# scripts/download_nsynth_train.py
#!/usr/bin/env python3
"""
Скрипт для загрузки и подготовки 5000 семплов из NSynth train split.
Оптимизирован для работы с большим архивом (~20GB).
"""

import os
import sys
import yaml
import json
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import warnings
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import tempfile

warnings.filterwarnings('ignore')

class NSynthTrainDownloader:
    def __init__(self, config_path="config/nsynth_train_config.yaml"):
        """Инициализация с конфигурационным файлом"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['nsynth_train']
        
        # Создаем директории
        self.archive_path = Path(self.config['archive_path'])
        self.extract_dir = Path(self.config['extract_dir'])
        self.output_dir = Path(self.config['output_dir'])
        
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройки выборки
        self.target_samples = self.config['target_samples']
        self.instrument_families = set(self.config['instrument_families'])
        self.pitch_ranges = self.config['pitch_ranges']
        
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        # Статистика
        self.stats = {
            'total_samples': 0,
            'downloaded': 0,
            'extracted': 0,
            'filtered': 0,
            'by_family': {},
            'by_pitch_range': {}
        }
    
    def download_with_progress(self, url, filepath, max_retries=3):
        """Скачивание файла с прогресс-баром и повторными попытками"""
        for attempt in range(max_retries):
            try:
                print(f"Попытка {attempt + 1}/{max_retries} скачивания архива...")
                
                # Класс для отображения прогресса
                class DownloadProgressBar(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)
                
                with DownloadProgressBar(unit='B', unit_scale=True,
                                        miniters=1, desc='Скачивание архива',
                                        ascii=True, ncols=100) as t:
                    urllib.request.urlretrieve(
                        url,
                        filename=filepath,
                        reporthook=t.update_to
                    )
                
                # Проверка целостности файла
                file_size = filepath.stat().st_size
                if file_size > 1000000:  # Минимум 1MB
                    print(f"✓ Скачан архив: {filepath}")
                    print(f"  Размер: {file_size / (1024**3):.2f} GB")
                    return True
                else:
                    print("✗ Файл слишком мал, возможно ошибка скачивания")
                    filepath.unlink(missing_ok=True)
                    
            except urllib.error.URLError as e:
                print(f"✗ Ошибка сети: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"  Ждем {wait_time} секунд перед повторной попыткой...")
                    time.sleep(wait_time)
                else:
                    print("✗ Достигнуто максимальное количество попыток")
            except Exception as e:
                print(f"✗ Неожиданная ошибка: {e}")
                return False
        
        return False
    
    def extract_metadata_streaming(self):
        """Извлечение метаданных из архива без распаковки всех файлов"""
        print("\nИзвлечение метаданных...")
        
        if not self.archive_path.exists():
            print("✗ Архив не найден!")
            return None
        
        try:
            metadata = {}
            total_samples = 0
            
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                # Ищем файл с метаданными
                metadata_members = [m for m in tar.getmembers() 
                                  if 'examples.json' in m.name]
                
                if not metadata_members:
                    print("✗ Файл метаданных не найден в архиве!")
                    return None
                
                # Извлекаем метаданные
                metadata_file = metadata_members[0]
                f = tar.extractfile(metadata_file)
                metadata = json.load(f)
                total_samples = len(metadata)
            
            print(f"✓ Загружено {total_samples:,} записей метаданных")
            return metadata
            
        except Exception as e:
            print(f"✗ Ошибка при извлечении метаданных: {e}")
            return None
    
    def analyze_metadata_distribution(self, metadata):
        """Анализ распределения данных по семействам и нотам"""
        print("\nАнализ распределения данных...")
        
        distribution = {
            'by_family': {},
            'by_pitch': {},
            'by_source': {}
        }
        
        # Собираем статистику
        for sample_id, sample_info in tqdm(metadata.items(), desc="Анализ"):
            family = sample_info.get('instrument_family_str', 'unknown')
            pitch = sample_info.get('pitch', -1)
            source = sample_info.get('instrument_source_str', 'unknown')
            velocity = sample_info.get('velocity', 64)
            
            # Фильтрация по качеству
            if velocity < self.config['min_velocity'] or velocity > self.config['max_velocity']:
                continue
            
            # По семейству
            if family not in distribution['by_family']:
                distribution['by_family'][family] = 0
            distribution['by_family'][family] += 1
            
            # По ноте (диапазону)
            pitch_range = self._get_pitch_range(pitch)
            if pitch_range not in distribution['by_pitch']:
                distribution['by_pitch'][pitch_range] = 0
            distribution['by_pitch'][pitch_range] += 1
            
            # По источнику
            if source not in distribution['by_source']:
                distribution['by_source'][source] = 0
            distribution['by_source'][source] += 1
        
        # Вывод статистики
        print("\nРаспределение по инструментальным семействам:")
        for family, count in sorted(distribution['by_family'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {family}: {count:,} ({count/len(metadata)*100:.1f}%)")
        
        print("\nРаспределение по диапазонам нот:")
        for pitch_range, count in distribution['by_pitch'].items():
            print(f"  {pitch_range}: {count:,}")
        
        print("\nРаспределение по источнику:")
        for source, count in distribution['by_source'].items():
            print(f"  {source}: {count:,}")
        
        return distribution
    
    def _get_pitch_range(self, pitch):
        """Определение диапазона ноты"""
        for range_name, (low, high) in self.pitch_ranges.items():
            if low <= pitch <= high:
                return range_name
        return "out_of_range"
    
    def select_samples_balanced(self, metadata, distribution):
        """Выбор семплов с балансировкой по семействам и нотам"""
        print("\nВыбор семплов с балансировкой...")
        
        # Целевое количество на семейство
        families_in_config = set(self.config['instrument_families'])
        available_families = set(distribution['by_family'].keys()) & families_in_config
        
        print(f"Доступные семейства из конфига: {', '.join(sorted(available_families))}")
        
        # Рассчитываем количество семплов на семейство
        samples_per_family = {}
        remaining_samples = self.target_samples
        
        # Первое приближение: пропорционально доступности
        for family in available_families:
            available = distribution['by_family'][family]
            # Ограничиваем максимальное количество на семейство
            max_per_family = min(available, self.config.get('samples_per_family', 500))
            samples_per_family[family] = max_per_family
        
        # Корректируем, если сумма превышает target_samples
        total_requested = sum(samples_per_family.values())
        if total_requested > self.target_samples:
            scale_factor = self.target_samples / total_requested
            for family in samples_per_family:
                samples_per_family[family] = int(samples_per_family[family] * scale_factor)
        
        # Балансировка по диапазонам нот внутри семейства
        selected_samples = []
        family_samples = {family: [] for family in available_families}
        
        # Сначала группируем все семплы по семейству
        print("\nГруппировка семплов по семействам...")
        grouped_by_family = {family: [] for family in available_families}
        
        for sample_id, sample_info in tqdm(metadata.items(), desc="Группировка"):
            family = sample_info.get('instrument_family_str', 'unknown')
            if family in available_families:
                grouped_by_family[family].append((sample_id, sample_info))
        
        # Выборка из каждой группы
        print("\nВыбор семплов из каждой группы...")
        for family, samples in grouped_by_family.items():
            if family not in samples_per_family:
                continue
            
            target_count = samples_per_family[family]
            if len(samples) < target_count:
                print(f"  Внимание: для {family} доступно только {len(samples)} из {target_count} запрошенных")
                target_count = len(samples)
            
            # Случайная выборка
            selected = random.sample(samples, target_count)
            
            # Сохраняем выбранные семплы
            for sample_id, sample_info in selected:
                selected_sample = {
                    'id': sample_id,
                    'instrument_family': family,
                    'pitch': sample_info.get('pitch', -1),
                    'instrument_source': sample_info.get('instrument_source_str', 'unknown'),
                    'velocity': sample_info.get('velocity', 64),
                    'qualities': sample_info.get('qualities_str', []),
                    'instrument': sample_info.get('instrument_str', -1),
                    'pitch_range': self._get_pitch_range(sample_info.get('pitch', -1))
                }
                selected_samples.append(selected_sample)
            
            # Обновляем статистику
            self.stats['by_family'][family] = target_count
        
        print(f"\nВыбрано {len(selected_samples)} семплов:")
        for family, count in self.stats['by_family'].items():
            print(f"  {family}: {count}")
        
        return selected_samples
    
    def extract_samples_batch(self, selected_samples, batch_size=1000):
        """Извлечение выбранных семплов батчами для экономии памяти"""
        print(f"\nИзвлечение {len(selected_samples)} аудиофайлов...")
        
        # Создаем структуру директорий
        for family in self.stats['by_family'].keys():
            family_dir = self.output_dir / family
            family_dir.mkdir(exist_ok=True)
        
        # Функция для извлечения одного семпла
        def extract_single(sample):
            try:
                sample_id = sample['id']
                family = sample['instrument_family']
                pitch = sample['pitch']
                
                # Путь к файлу в архиве
                audio_path_in_archive = f"nsynth-train/audio/{sample_id}.wav"
                
                with tarfile.open(self.archive_path, 'r:gz') as tar:
                    # Извлекаем файл во временную директорию
                    tar.extract(audio_path_in_archive, path=self.extract_dir)
                    
                    # Перемещаем в целевую директорию
                    src_path = self.extract_dir / audio_path_in_archive
                    dst_path = self.output_dir / family / f"{sample_id}_pitch{pitch}.wav"
                    
                    shutil.move(str(src_path), str(dst_path))
                    
                    return True, sample_id
                    
            except Exception as e:
                return False, f"{sample_id}: {e}"
        
        # Извлекаем батчами
        extracted_count = 0
        failed_samples = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Создаем задачи для каждого семпла
            futures = []
            for sample in selected_samples:
                future = executor.submit(extract_single, sample)
                futures.append(future)
            
            # Обрабатываем результаты с прогресс-баром
            with tqdm(total=len(selected_samples), desc="Извлечение файлов") as pbar:
                for future in as_completed(futures):
                    success, result = future.result()
                    if success:
                        extracted_count += 1
                    else:
                        failed_samples.append(result)
                    pbar.update(1)
        
        # Очищаем временные файлы
        temp_dir = self.extract_dir / "nsynth-train"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        print(f"\n✓ Извлечено {extracted_count} аудиофайлов")
        if failed_samples:
            print(f"✗ Не удалось извлечь {len(failed_samples)} файлов:")
            for fail in failed_samples[:10]:  # Показываем первые 10 ошибок
                print(f"  {fail}")
            if len(failed_samples) > 10:
                print(f"  ... и еще {len(failed_samples) - 10} ошибок")
        
        return extracted_count
    
    def create_metadata_files(self, selected_samples):
        """Создание файлов с метаданными и маппингом атрибутов"""
        print("\nСоздание файлов метаданных...")
        
        # Создаем DataFrame с метаданными
        metadata_list = []
        for sample in selected_samples:
            family = sample['instrument_family']
            sample_id = sample['id']
            pitch = sample['pitch']
            
            # Проверяем, существует ли файл
            file_path = f"{family}/{sample_id}_pitch{pitch}.wav"
            full_path = self.output_dir / file_path
            
            if full_path.exists():
                metadata = {
                    'file_path': file_path,
                    'file_name': f"{sample_id}_pitch{pitch}.wav",
                    'instrument_family': family,
                    'instrument_source': sample['instrument_source'],
                    'pitch': pitch,
                    'velocity': sample['velocity'],
                    'qualities': ','.join(map(str, sample['qualities'])),
                    'instrument_id': sample['instrument'],
                    'pitch_range': sample['pitch_range']
                }
                metadata_list.append(metadata)
        
        df = pd.DataFrame(metadata_list)
        
        # Сохраняем в CSV
        csv_path = self.output_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Метаданные сохранены в: {csv_path}")
        print(f"  Количество записей: {len(df)}")
        print(f"  Колонки: {list(df.columns)}")
        
        # Создаем маппинг атрибутов
        attribute_map = {
            'instrument_family': {
                'bass': 'bass',
                'brass': 'brass',
                'keyboard': 'keyboard',
                'mallet': 'percussion',
                'synth_lead': 'synth',
                'string': 'bowed_string',
                'vocal': 'vocal',
                'guitar': 'guitar',
                'organ': 'keyboard',
                'reed': 'reed',
                'flute': 'flute'
            },
            'instrument_source': {
                'acoustic': 'acoustic',
                'electronic': 'electronic',
                'synthetic': 'synthetic'
            },
            'pitch_range': self.pitch_ranges
        }
        
        mapping_path = self.output_dir / "attribute_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(attribute_map, f, indent=2)
        
        print(f"✓ Маппинг атрибутов сохранен в: {mapping_path}")
        
        # Сохраняем статистику
        stats_path = self.output_dir / "sampling_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        return df
    
    def verify_dataset(self):
        """Проверка целостности созданного датасета"""
        print("\nПроверка целостности датасета...")
        
        # Проверяем наличие файлов
        wav_files = list(self.output_dir.rglob("*.wav"))
        csv_files = list(self.output_dir.glob("*.csv"))
        json_files = list(self.output_dir.glob("*.json"))
        
        print(f"  Найдено .wav файлов: {len(wav_files)}")
        print(f"  Найдено .csv файлов: {len(csv_files)}")
        print(f"  Найдено .json файлов: {len(json_files)}")
        
        # Проверяем размеры файлов
        total_size = sum(f.stat().st_size for f in wav_files)
        avg_size = total_size / len(wav_files) if wav_files else 0
        
        print(f"  Общий размер аудио: {total_size / (1024**3):.2f} GB")
        print(f"  Средний размер файла: {avg_size / 1024:.1f} KB")
        
        # Проверяем CSV файл
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                print(f"  Записей в CSV: {len(df)}")
                print(f"  Соответствие файлам: {'✓' if len(df) == len(wav_files) else '✗'}")
            except Exception as e:
                print(f"  Ошибка чтения CSV: {e}")
        
        return len(wav_files) > 0
    
    def extract_fast_single_pass(self, selected_samples):
        """
        Быстрое извлечение файлов за один проход по архиву.
        Это самый эффективный способ для большого количества файлов.
        """
        print(f"\nБыстрое извлечение {len(selected_samples)} аудиофайлов...")
        print("Этот процесс может занять 10-30 минут в зависимости от скорости диска.")
        
        start_time = time.time()
        
        # Создаем множество нужных путей в архиве для быстрого поиска
        needed_paths = set()
        sample_info = {}  # Для сопоставления пути с информацией о семпле
        
        for sample in selected_samples:
            sample_id = sample['id']
            family = sample['instrument_family']
            pitch = sample['pitch']
            
            # Путь к файлу в архиве
            audio_path_in_archive = f"nsynth-train/audio/{sample_id}.wav"
            needed_paths.add(audio_path_in_archive)
            
            # Сохраняем информацию для последующего использования
            sample_info[audio_path_in_archive] = {
                'family': family,
                'pitch': pitch,
                'sample_id': sample_id
            }
        
        print(f"Подготовлено {len(needed_paths)} путей для извлечения")
        
        # Создаем директории для каждого семейства
        for family in set([s['family'] for s in sample_info.values()]):
            family_dir = self.output_dir / family
            family_dir.mkdir(exist_ok=True)
        
        extracted_count = 0
        skipped_count = 0
        
        try:
            # Открываем архив один раз и проходимся по всем файлам
            with tarfile.open(self.archive_path, 'r:gz') as tar:
                # Получаем список всех членов архива
                total_members = len(tar.getmembers())
                print(f"Всего файлов в архиве: {total_members:,}")
                
                # Используем tqdm для прогресса
                with tqdm(total=len(needed_paths), desc="Извлечение файлов", 
                         unit="файл", ncols=100) as pbar:
                    
                    # Итерируемся по всем файлам в архиве
                    for member in tar:
                        if member.name in needed_paths:
                            try:
                                # Извлекаем файл
                                tar.extract(member, path=self.extract_dir)
                                
                                # Получаем информацию о семпле
                                info = sample_info[member.name]
                                family = info['family']
                                pitch = info['pitch']
                                sample_id = info['sample_id']
                                
                                # Определяем исходный и целевой пути
                                src_path = self.extract_dir / member.name
                                dst_path = self.output_dir / family / f"{sample_id}_pitch{pitch}.wav"
                                
                                # Перемещаем файл в целевую директорию
                                if src_path.exists():
                                    shutil.move(str(src_path), str(dst_path))
                                    extracted_count += 1
                                else:
                                    skipped_count += 1
                                    print(f"\nПредупреждение: файл не найден после извлечения: {member.name}")
                                
                                # Обновляем прогресс
                                pbar.update(1)
                                
                                # Очищаем извлеченные файлы из множества для экономии памяти
                                needed_paths.remove(member.name)
                                
                                # Если все файлы извлечены, выходим раньше
                                if not needed_paths:
                                    break
                                    
                            except Exception as e:
                                print(f"\nОшибка при извлечении {member.name}: {e}")
                                skipped_count += 1
                                pbar.update(1)
        
        except Exception as e:
            print(f"\nКритическая ошибка при работе с архивом: {e}")
        
        # Очищаем временную директорию
        temp_dir = self.extract_dir / "nsynth-train"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Игнорируем ошибки очистки
        
        elapsed_time = time.time() - start_time
        self.stats['extraction_time'] = elapsed_time
        
        print(f"\n✓ Извлечено {extracted_count} аудиофайлов за {elapsed_time/60:.1f} минут")
        if skipped_count > 0:
            print(f"✗ Пропущено {skipped_count} файлов")
        
        return extracted_count
    
    def extract_using_tar_command(self, selected_samples):
        """
        Использование системной команды tar для максимальной скорости.
        Это самый быстрый способ на Linux/macOS.
        """
        print(f"\nИспользование системной команды tar для извлечения {len(selected_samples)} файлов...")
        
        # Создаем временный файл со списком файлов для извлечения
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for sample in selected_samples:
                sample_id = sample['id']
                # Записываем путь к файлу в архиве
                f.write(f"nsynth-train/audio/{sample_id}.wav\n")
            list_file = f.name
        
        try:
            # Создаем директории для каждого семейства
            for sample in selected_samples:
                family_dir = self.output_dir / sample['instrument_family']
                family_dir.mkdir(exist_ok=True)
            
            # Команда для извлечения только указанных файлов
            cmd = [
                'tar', '-xzf', str(self.archive_path),
                '-C', str(self.extract_dir),
                '--files-from', list_file
            ]
            
            print(f"Выполнение команды: {' '.join(cmd)}")
            
            # Запускаем команду
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"Ошибка выполнения tar: {result.stderr}")
                return 0
            
            # Перемещаем извлеченные файлы в целевую структуру
            extracted_count = 0
            for sample in tqdm(selected_samples, desc="Перемещение файлов"):
                sample_id = sample['id']
                family = sample['instrument_family']
                pitch = sample['pitch']
                
                src_path = self.extract_dir / f"nsynth-train/audio/{sample_id}.wav"
                dst_path = self.output_dir / family / f"{sample_id}_pitch{pitch}.wav"
                
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                    extracted_count += 1
            
            # Очищаем временные файлы
            temp_dir = self.extract_dir / "nsynth-train"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            print(f"✓ Извлечено {extracted_count} файлов за {elapsed_time:.1f} секунд")
            return extracted_count
            
        except Exception as e:
            print(f"Ошибка при использовании tar: {e}")
            # Возвращаемся к Python методу
            return self.extract_fast_single_pass(selected_samples)
        finally:
            # Удаляем временный файл
            if os.path.exists(list_file):
                os.unlink(list_file)
    
    def extract_batch_optimized(self, selected_samples, method='auto'):
        """
        Оптимизированное извлечение файлов.
        method: 'auto', 'python', 'tar'
        """
        print(f"\nОптимизированное извлечение {len(selected_samples)} файлов...")
        
        # Проверяем доступность системного tar
        if method == 'auto':
            try:
                # Проверяем, доступна ли команда tar
                subprocess.run(['tar', '--version'], capture_output=True, check=True)
                print("Обнаружен системный tar, используем его для максимальной скорости")
                method = 'tar'
            except:
                print("Системный tar не найден, используем Python реализацию")
                method = 'python'
        
        if method == 'tar':
            return self.extract_using_tar_command(selected_samples)
        else:
            return self.extract_fast_single_pass(selected_samples)
    
    def run_optimized(self, force_redownload=False):
        """Оптимизированная версия основного метода"""
        print("=" * 70)
        print("ОПТИМИЗИРОВАННАЯ ЗАГРУЗКА 5000 СЕМПЛОВ")
        print("=" * 70)
        
        # Проверяем существование датасета
        if self.output_dir.exists() and len(list(self.output_dir.rglob("*.wav"))) >= 3000:
            print(f"✓ Датасет уже существует в {self.output_dir}")
            return True
        
        # Шаг 1: Скачивание архива
        if not self.archive_path.exists() or force_redownload:
            print("\n1. СКАЧИВАНИЕ АРХИВА")
            # Используем существующий метод скачивания
            if not self.download_with_progress(
                self.config['dataset_url'],
                self.archive_path,
                max_retries=self.config['max_download_retries']
            ):
                return False
        else:
            archive_size = self.archive_path.stat().st_size / (1024**3)
            print(f"✓ Архив уже скачан: {archive_size:.2f} GB")
        
        # Шаг 2: Чтение метаданных и выбор семплов
        print("\n2. ОБРАБОТКА МЕТАДАННЫХ")
        
        # Используем быстрый метод чтения метаданных
        metadata = self.extract_metadata_streaming()
        if not metadata:
            return False
        
        # Шаг 3: Анализ распределения
        distribution = self.analyze_metadata_distribution(metadata)

        # Выбираем семплы
        selected_samples = self.select_samples_balanced(metadata, distribution)
        
        if len(selected_samples) < 3000:
            print(f"✗ Недостаточно семплов: {len(selected_samples)}")
            return False
        
        # Шаг 3: Быстрое извлечение
        print("\n3. БЫСТРОЕ ИЗВЛЕЧЕНИЕ ФАЙЛОВ")
        extracted_count = self.extract_batch_optimized(selected_samples, method='auto')
        
        if extracted_count < 2000:
            print(f"✗ Извлечено слишком мало файлов: {extracted_count}")
            return False
        
        # Шаг 4: Создание метаданных
        print("\n4. СОЗДАНИЕ МЕТАДАННЫХ")
        self.create_metadata_files(selected_samples)
        
        print("\n" + "=" * 70)
        print("ГОТОВО!")
        print(f"Создан датасет с {extracted_count} семплами")
        print(f"Время извлечения: {self.stats['extraction_time']/60:.1f} минут")
        
        return True

def main():
    """Точка входа для оптимизированной версии"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Оптимизированная загрузка 5000 семплов из NSynth'
    )
    parser.add_argument('--config', default='config/nsynth_train_config.yaml')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--method', choices=['auto', 'python', 'tar'], default='auto')
    
    args = parser.parse_args()
    
    downloader = NSynthTrainDownloader(args.config)
    success = downloader.run_optimized(force_redownload=args.force)
    
    if not success:
        print("\n✗ Ошибка при загрузке")
        sys.exit(1)

if __name__ == "__main__":
    main()