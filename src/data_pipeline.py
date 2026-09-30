# src/data_pipeline.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.preprocessing import OneHotEncoder
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class AudioValidator:
    """Класс для валидации аудиофайлов"""
    
    def __init__(self, config):
        self.config = config
        self.sr = self.config['target_sr'] if self.config['resample'] else self.config['sr']
        
    def validate_file(self, file_path):
        """Полная валидация аудиофайла"""
        validation_result = {
            'file_path': str(file_path),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Проверка существования файла
            if not file_path.exists():
                validation_result['is_valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Получение базовой информации
            file_info = sf.info(file_path)
            validation_result['metadata'].update({
                'duration': file_info.duration,
                'sr': file_info.samplerate,
                'channels': file_info.channels,
                'subtype': file_info.subtype
            })
            
            # 1. Проверка длительности
            if file_info.duration < self.config['min_duration']:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"Duration too short: {file_info.duration:.2f}s < {self.config['min_duration']}s"
                )
            
            # 2. Проверка частоты дискретизации
            if file_info.samplerate < self.config['sr']:
                validation_result['warnings'].append(
                    f"Sample rate lower than expected: {file_info.samplerate} < {self.config['sr']}"
                )
            
            # 3. Проверка моно/стерео
            if file_info.channels > 1 and self.config['mono']:
                validation_result['warnings'].append(
                    f"File has {file_info.channels} channels, will be converted to mono"
                )
            
            # 4. Загрузка части файла для проверки клиппинга
            test_duration = min(1.0, file_info.duration)  # Проверяем только 1 секунду для скорости
            y, sr = librosa.load(
                file_path,
                sr=None,
                mono=False,
                duration=test_duration,
                offset=file_info.duration/2 - test_duration/2  # Проверяем середину
            )
            
            # Проверка клиппинга
            if len(y.shape) > 1:  # Если стерео
                y_mono = librosa.to_mono(y)
            else:
                y_mono = y
            
            max_amplitude = np.max(np.abs(y_mono))
            if max_amplitude > self.config['max_clipping']:
                validation_result['warnings'].append(
                    f"Possible clipping detected: max amplitude = {max_amplitude:.3f}"
                )
            
            # 5. Проверка тишины (слишком тихий файл)
            rms = np.sqrt(np.mean(y_mono**2))
            if rms < 0.001:  # Порог для тихого файла
                validation_result['warnings'].append(
                    f"File might be too quiet: RMS = {rms:.6f}"
                )
            
            # Проверка на NaN/Inf
            if np.any(np.isnan(y_mono)) or np.any(np.isinf(y_mono)):
                validation_result['is_valid'] = False
                validation_result['errors'].append("File contains NaN or Inf values")
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result

class AudioProcessor:
    """Класс для обработки аудио"""
    
    def __init__(self, config):
        self.config = config
        
        # Определяем целевую частоту дискретизации
        if self.config['resample']:
            self.target_sr = self.config['target_sr']
        else:
            self.target_sr = self.config['sr']
        
        # Количество семплов для сегмента
        self.segment_samples = int(self.target_sr * self.config['duration'])
    
    def load_and_preprocess(self, file_path):
        """Загрузка и предобработка аудиофайла"""
        try:
            # Загрузка с исходной частотой дискретизации
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Конвертация в моно если нужно
            if len(y.shape) > 1 and self.config['mono']:
                y = librosa.to_mono(y)
            elif len(y.shape) == 1:
                y = y.reshape(1, -1)
            
            # Ресемплирование если нужно
            if self.config['resample'] and sr != self.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            elif not self.config['resample']:
                self.target_sr = sr
            
            # Извлечение центрального сегмента
            y = self.extract_central_segment(y)
            
            # Нормализация громкости
            y = self.normalize_volume(y)
            
            return y, self.target_sr
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def extract_central_segment(self, audio):
        """Извлечение центрального сегмента фиксированной длины"""
        # audio shape: (channels, samples) или (samples,)
        if len(audio.shape) == 1:
            total_samples = len(audio)
        else:
            total_samples = audio.shape[1]
        
        # Если аудио короче целевой длительности - повторяем
        if total_samples < self.segment_samples:
            repeats = int(np.ceil(self.segment_samples / total_samples))
            if len(audio.shape) == 1:
                audio = np.tile(audio, repeats)
            else:
                audio = np.tile(audio, (1, repeats))
            total_samples = audio.shape[1] if len(audio.shape) > 1 else len(audio)
        
        # Извлечение центрального сегмента
        start = (total_samples - self.segment_samples) // 2
        end = start + self.segment_samples
        
        if len(audio.shape) == 1:
            segment = audio[start:end]
        else:
            segment = audio[:, start:end]
        
        return segment
    
    def normalize_volume(self, audio):
        """Нормализация громкости по RMS"""
        if not self.config['normalize']:
            return audio
        
        # Вычисление RMS
        if len(audio.shape) == 1:
            rms = np.sqrt(np.mean(audio**2))
        else:
            rms = np.sqrt(np.mean(audio**2, axis=1))
            rms = np.mean(rms)  # Среднее по каналам
        
        # Нормализация
        if rms > 0:
            target_rms = self.config['target_rms']
            gain = target_rms / rms
            
            # Ограничение максимального усиления
            max_gain = 10.0  # Не усиливаем более чем в 10 раз
            gain = min(gain, max_gain)
            
            audio = audio * gain
        
        # Ограничение амплитуды
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def compute_melspectrogram(self, audio, sr):
        """Вычисление мел-спектрограммы"""
        # Убедимся, что audio моно
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Вычисление STFT
        stft = librosa.stft(
            audio,
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['n_fft'],
            window='hann'
        )
        
        # Мел-спектрограмма
        mel_basis = librosa.filters.mel(
            sr=sr,
            n_fft=self.config['n_fft'],
            n_mels=self.config['n_mels'],
            fmax=self.config.get('fmax', sr//2)
        )
        
        mel_spec = np.dot(mel_basis, np.abs(stft))
        
        # Логарифмическое масштабирование
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Нормализация к [0, 1]
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-8
        )
        
        return mel_spec_normalized

class MetadataProcessor:
    """Класс для обработки метаданных"""
    
    def __init__(self, config, attribute_config_path=None):
        self.config = config
        self.attribute_config = self.config['attributes']
        
        # Создаем энкодеры для атрибутов
        self.encoders = {}
        self.attribute_values = {}
        
        for attr in self.attribute_config:
            attr_name = attr['name']
            if 'values' in attr:
                self.attribute_values[attr_name] = attr['values']
                # Создаем OneHotEncoder для этого атрибута
                encoder = OneHotEncoder(sparse_output=False, categories=[attr['values']])
                self.encoders[attr_name] = encoder
            elif 'mapping' in attr:
                # Для атрибутов с маппингом используем значения из mapping
                values = list(attr['mapping'].keys())
                self.attribute_values[attr_name] = values
                encoder = OneHotEncoder(sparse_output=False, categories=[values])
                self.encoders[attr_name] = encoder
    
    def extract_attributes_from_metadata(self, metadata_row):
        """Извлечение атрибутов из строки метаданных"""
        attributes = {}
        
        for attr in self.attribute_config:
            attr_name = attr['name']
            source_column = attr.get('source_column')
            
            if source_column and source_column in metadata_row:
                source_value = metadata_row[source_column]
                
                if 'mapping' in attr:
                    # Применяем маппинг
                    mapped_value = attr['default']
                    for key, ranges in attr['mapping'].items():
                        if isinstance(ranges, list) and len(ranges) == 2:
                            if ranges[0] <= source_value <= ranges[1]:
                                mapped_value = key
                                break
                    attributes[attr_name] = mapped_value
                else:
                    attributes[attr_name] = source_value
            else:
                # Используем значение по умолчанию
                attributes[attr_name] = attr.get('default', 'unknown')
        
        return attributes
    
    def encode_attributes(self, attributes_dict):
        """One-hot кодирование атрибутов"""
        encoded_vectors = []
        encoded_info = {}
        
        for attr_name, encoder in self.encoders.items():
            if attr_name in attributes_dict:
                value = attributes_dict[attr_name]
                
                # Если значение не в списке допустимых, используем первое
                if value not in self.attribute_values[attr_name]:
                    value = self.attribute_values[attr_name][0]
                
                # One-hot кодирование
                encoded = encoder.fit_transform([[value]])
                encoded_vectors.append(encoded[0])
                
                # Сохраняем информацию для декодирования
                encoded_info[attr_name] = {
                    'value': value,
                    'vector': encoded[0].tolist(),
                    'index': np.argmax(encoded[0])
                }
        
        # Объединяем все векторы
        if encoded_vectors:
            full_vector = np.concatenate(encoded_vectors)
            encoded_info['full_vector'] = full_vector.tolist()
        
        return encoded_info

class DataPipeline:
    """Основной класс пайплайна обработки данных"""
    
    def __init__(self, config_path="config/data_config.yaml"):
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data_pipeline']
        
        # Инициализация компонентов
        self.validator = AudioValidator(self.config)
        self.processor = AudioProcessor(self.config)
        self.metadata_processor = MetadataProcessor(self.config)
        
        # Для статистики
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'processing_errors': 0
        }
        
        # Временная метка для логов
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def process_dataset(self, input_dir, metadata_csv, output_dir):
        """Основной метод обработки датасета"""
        print("=" * 60)
        print("ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ ДАННЫХ")
        print("=" * 60)
        
        # Создание директорий
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Загрузка метаданных
        print("Загрузка метаданных...")
        metadata_df = pd.read_csv(metadata_csv)
        print(f"Загружено {len(metadata_df)} записей метаданных")
        
        # Создание лог-файлов
        log_dir = output_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        validation_log = []
        processing_log = []
        
        # Списки для результатов
        all_specs = []
        all_metadata = []
        
        # Проход по всем файлам
        print("\nОбработка файлов...")
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Файлы"):
            self.stats['total_files'] += 1
            
            file_path = input_path / row['file_path']
            
            # Шаг 1: Валидация
            validation_result = self.validator.validate_file(file_path)
            validation_log.append(validation_result)
            
            if not validation_result['is_valid']:
                self.stats['invalid_files'] += 1
                print(f"\nПропущен {file_path}: {validation_result['errors']}")
                continue
            
            self.stats['valid_files'] += 1
            
            # Шаг 2: Обработка аудио
            try:
                # Загрузка и предобработка
                audio, sr = self.processor.load_and_preprocess(file_path)
                
                if audio is None:
                    self.stats['processing_errors'] += 1
                    processing_log.append({
                        'file': str(file_path),
                        'status': 'ERROR',
                        'message': 'Failed to load/process audio'
                    })
                    continue
                
                # Вычисление спектрограммы
                mel_spec = self.processor.compute_melspectrogram(audio, sr)
                
                # Шаг 3: Обработка метаданных
                attributes = self.metadata_processor.extract_attributes_from_metadata(row)
                encoded_attrs = self.metadata_processor.encode_attributes(attributes)
                
                # Сохранение информации о семпле (преобразуем NumPy типы)
                sample_metadata = {
                    'original_path': str(file_path),
                    'original_filename': file_path.name,
                    'instrument_family': str(row.get('instrument_family', 'unknown')),
                    'instrument_source': str(row.get('instrument_source', 'unknown')),
                    'pitch': int(row.get('pitch', 0)) if isinstance(row.get('pitch', 0), (np.integer, int)) else row.get('pitch', 0),
                    'velocity': int(row.get('velocity', 64)) if isinstance(row.get('velocity', 64), (np.integer, int)) else row.get('velocity', 64),
                    'spec_shape': [int(mel_spec.shape[0]), int(mel_spec.shape[1])],
                    'sample_rate': int(sr) if isinstance(sr, (np.integer, int)) else sr,
                    'duration_seconds': float(self.config['duration']),
                }
                
                # Добавляем атрибуты (уже строки)
                for key, value in attributes.items():
                    sample_metadata[key] = str(value)
                
                # Преобразуем encoded_attrs для сериализации
                encoded_attrs_serializable = {}
                for attr_name, attr_data in encoded_attrs.items():
                    if isinstance(attr_data, dict):
                        encoded_attrs_serializable[attr_name] = {}
                        for k, v in attr_data.items():
                            if isinstance(v, (np.ndarray, list)):
                                encoded_attrs_serializable[attr_name][k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v] if k == 'vector' else v
                            elif isinstance(v, (np.integer, int)):
                                encoded_attrs_serializable[attr_name][k] = int(v)
                            elif isinstance(v, (np.floating, float)):
                                encoded_attrs_serializable[attr_name][k] = float(v)
                            else:
                                encoded_attrs_serializable[attr_name][k] = v
                    else:
                        encoded_attrs_serializable[attr_name] = attr_data
                
                sample_metadata['encoded_attributes'] = encoded_attrs_serializable
                
                # Добавление в коллекции
                all_specs.append(mel_spec)
                all_metadata.append(sample_metadata)
                
                processing_log.append({
                    'file': str(file_path),
                    'status': 'SUCCESS',
                    'spec_shape': f"{mel_spec.shape[0]}x{mel_spec.shape[1]}",
                    'sample_rate': int(sr)
                })
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                processing_log.append({
                    'file': str(file_path),
                    'status': 'ERROR',
                    'message': str(e)
                })
                print(f"\nОшибка обработки {file_path}: {e}")
        
        # Сохранение результатов
        print("\nСохранение результатов...")
        
        if all_specs:
            # Преобразование спектрограмм в numpy массив
            specs_array = np.stack(all_specs, axis=0)
            
            # Добавление размерности канала для CNN
            specs_array = specs_array[:, np.newaxis, :, :]  # (n_samples, 1, n_mels, time)
            
            # Сохранение спектрограмм
            specs_path = output_path / "melspectrograms.npy"
            np.save(specs_path, specs_array)
            print(f"Спектрограммы сохранены: {specs_path} ({specs_array.shape})")
            
            # Сохранение метаданных
            metadata_save_path = output_path / "processed_metadata.json"
            
            # Функция для рекурсивной конвертации NumPy типов
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            # Конвертируем все метаданные
            all_metadata_serializable = convert_numpy_types(all_metadata)
            
            # Сохраняем в JSON для сохранения структуры
            with open(metadata_save_path, 'w', encoding='utf-8') as f:
                json.dump(all_metadata_serializable, f, indent=2, ensure_ascii=False)
            
            # Также сохраняем в CSV для удобства просмотра
            simplified_metadata = []
            for meta in all_metadata:
                simplified = {
                    'filename': meta['original_filename'],
                    'instrument_family': meta['instrument_family'],
                    'instrument_source': meta['instrument_source'],
                    'pitch': meta['pitch'],
                    'velocity': meta['velocity'],
                    'brightness': meta.get('brightness', 'unknown'),
                    'attack_type': meta.get('attack_type', 'unknown'),
                    'spec_height': meta['spec_shape'][0],
                    'spec_width': meta['spec_shape'][1]
                }
                simplified_metadata.append(simplified)
            
            metadata_csv_path = output_path / "metadata_summary.csv"
            pd.DataFrame(simplified_metadata).to_csv(metadata_csv_path, index=False)
            print(f"Метаданные (CSV) сохранены: {metadata_csv_path}")
        
        # Сохранение логов
        print("\nСохранение логов...")
        
        # Лог валидации
        validation_df = pd.DataFrame(validation_log)
        validation_log_path = log_dir / f"validation_log_{self.timestamp}.csv"
        validation_df.to_csv(validation_log_path, index=False)
        
        # Лог обработки
        processing_df = pd.DataFrame(processing_log)
        processing_log_path = log_dir / f"processing_log_{self.timestamp}.csv"
        processing_df.to_csv(processing_log_path, index=False)
        
        # Сохранение конфигурации
        config_save_path = output_path / "pipeline_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump({'data_pipeline': self.config}, f)
        
        # Вывод статистики
        self.print_statistics()
        
        # Визуализация результатов
        self.visualize_results(output_path, all_specs[:5] if all_specs else [])
        
        return specs_array, all_metadata
        
        
    def print_statistics(self):
        """Вывод статистики обработки"""
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ОБРАБОТКИ")
        print("=" * 60)
        print(f"Всего файлов: {self.stats['total_files']}")
        print(f"Валидных файлов: {self.stats['valid_files']}")
        print(f"Невалидных файлов: {self.stats['invalid_files']}")
        print(f"Ошибок обработки: {self.stats['processing_errors']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['valid_files'] / self.stats['total_files']) * 100
            print(f"Успешная обработка: {success_rate:.1f}%")
    
    def visualize_results(self, output_path, sample_specs):
        """Создание визуализаций для проверки"""
        if not sample_specs:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            vis_dir = output_path / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # Визуализация нескольких спектрограмм
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, spec in enumerate(sample_specs[:6]):
                ax = axes[i]
                img = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f"Спектрограмма {i+1}")
                ax.set_xlabel("Время (кадры)")
                ax.set_ylabel("Мел-банки")
                plt.colorbar(img, ax=ax)
            
            plt.tight_layout()
            vis_path = vis_dir / f"sample_spectrograms_{self.timestamp}.png"
            plt.savefig(vis_path, dpi=150)
            plt.close()
            
            print(f"\nВизуализации сохранены в: {vis_dir}")
            
        except ImportError:
            print("\nMatplotlib не установлен, пропускаем визуализацию")
        except Exception as e:
            print(f"\nОшибка при создании визуализаций: {e}")

def main():
    """Точка входа для обработки данных"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Пайплайн обработки аудиоданных')
    parser.add_argument('--input', required=True, help='Директория с исходными аудиофайлами')
    parser.add_argument('--metadata', required=True, help='CSV файл с метаданными')
    parser.add_argument('--output', required=True, help='Директория для сохранения результатов')
    parser.add_argument('--config', default='config/data_config.yaml', 
                       help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    # Проверка путей
    input_path = Path(args.input)
    metadata_path = Path(args.metadata)
    
    if not input_path.exists():
        print(f"Ошибка: Директория {input_path} не существует")
        return
    
    if not metadata_path.exists():
        print(f"Ошибка: Файл метаданных {metadata_path} не существует")
        return
    
    # Запуск пайплайна
    try:
        pipeline = DataPipeline(args.config)
        specs, metadata = pipeline.process_dataset(
            input_dir=args.input,
            metadata_csv=args.metadata,
            output_dir=args.output
        )
        
        if specs is not None:
            print("\n" + "=" * 60)
            print("ОБРАБОТКА УСПЕШНО ЗАВЕРШЕНА!")
            print("=" * 60)
            print(f"Создано {len(metadata)} семплов")
            print(f"Размерность данных: {specs.shape}")
            print(f"Результаты сохранены в: {args.output}")
        else:
            print("\nОбработка не удалась. Проверьте логи.")
            
    except Exception as e:
        print(f"\nКритическая ошибка при выполнении пайплайна: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()