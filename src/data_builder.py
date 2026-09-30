# src/data_builder.py
import os
import yaml
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataBuilder:
    def __init__(self, config_path="config/data_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data_pipeline']
        
        self.sr = self.config['sr']
        self.duration = self.config['duration']
        self.n_samples = int(self.sr * self.duration)
        self.n_mels = self.config['n_mels']
        self.hop_length = self.config['hop_length']
        self.n_fft = self.config['n_fft']
        
        # Создадим маппинг для one-hot кодирования атрибутов
        self.attr_encoders = {}
        for attr in self.config['attributes']:
            values = attr['values']
            self.attr_encoders[attr['name']] = {
                val: idx for idx, val in enumerate(values)
            }
    
    def validate_audio(self, audio_path):
        """Проверка аудиофайла на соответствие требованиям"""
        try:
            # Проверка формата
            if not audio_path.endswith('.wav'):
                return False, "Not a WAV file"
            
            # Загрузка метаданных
            info = librosa.util.valid_audio(audio_path)
            if info is False:
                return False, "Invalid audio file"
            
            # Проверка моно/стерео
            y, sr = librosa.load(audio_path, sr=None, mono=False, duration=0.1)
            if len(y.shape) > 1 and y.shape[0] > 1 and self.config['mono']:
                return False, "Not mono"
            
            # Проверка частоты дискретизации
            if sr < self.sr:
                return False, f"Sample rate too low: {sr} < {self.sr}"
            
            # Проверка клиппинга
            y_full, _ = librosa.load(audio_path, sr=sr, mono=True)
            if np.max(np.abs(y_full)) > 0.99:
                return False, "Possible clipping detected"
                
            return True, "OK"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def extract_segment(self, audio_path):
        """Извлечение центрального сегмента фиксированной длины"""
        try:
            # Загрузка аудио
            y, sr = librosa.load(audio_path, sr=self.sr, mono=self.config['mono'])
            
            # Если аудио короче целевой длительности - повторяем
            if len(y) < self.n_samples:
                repeats = int(np.ceil(self.n_samples / len(y)))
                y = np.tile(y, repeats)
            
            # Выбор центрального сегмента
            if len(y) > self.n_samples:
                start = (len(y) - self.n_samples) // 2
                y = y[start:start + self.n_samples]
            elif len(y) < self.n_samples:
                # Паддинг если всё ещё меньше (маловероятно)
                pad_len = self.n_samples - len(y)
                y = np.pad(y, (0, pad_len), mode='constant')
            
            # Нормализация громкости
            if self.config['normalize']:
                rms = np.sqrt(np.mean(y**2))
                if rms > 0:
                    y = y * (self.config['target_rms'] / rms)
                y = np.clip(y, -1.0, 1.0)
            
            return y
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def compute_melspectrogram(self, audio):
        """Вычисление мел-спектрограммы"""
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Мел-спектрограмма в dB
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(stft),
            sr=self.sr,
            n_mels=self.n_mels,
            fmax=self.sr//2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Нормализация от 0 до 1
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_normalized
    
    def parse_metadata(self, audio_path):
        """Парсинг метаданных из структуры папок и имени файла"""
        # Пример: /path/to/data/violin/bright/sample1.wav
        path_parts = Path(audio_path).parts
        
        # Базовые метаданные
        metadata = {
            'file_path': audio_path,
            'filename': Path(audio_path).name,
            'instrument': path_parts[-2] if len(path_parts) > 1 else 'unknown'
        }
        
        # Здесь должна быть логика извлечения атрибутов
        # Пока заглушка - случайные значения для теста
        metadata['instrument_type'] = np.random.choice(['string', 'woodwind', 'synth'])
        metadata['attack_type'] = np.random.choice(['fast', 'medium', 'slow'])
        metadata['brightness'] = np.random.choice(['dark', 'medium', 'bright'])
        
        # One-hot кодирование
        one_hot = []
        for attr_name in self.attr_encoders:
            val = metadata.get(attr_name, 'unknown')
            encoder = self.attr_encoders[attr_name]
            vec = np.zeros(len(encoder), dtype=np.float32)
            if val in encoder:
                vec[encoder[val]] = 1.0
            one_hot.extend(vec.tolist())
        
        metadata['one_hot_vector'] = one_hot
        
        return metadata
    
    def build_dataset(self, input_dir, output_dir):
        """Основной метод построения датасета"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Логи
        log_file = output_path / "processing_log.csv"
        log_data = []
        
        # Списки для результатов
        all_specs = []
        all_metadata = []
        
        # Поиск всех аудиофайлов
        audio_files = list(input_path.rglob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        for audio_file in tqdm(audio_files, desc="Processing audio"):
            # Валидация
            is_valid, message = self.validate_audio(str(audio_file))
            log_data.append({
                'file': str(audio_file),
                'status': 'VALID' if is_valid else 'INVALID',
                'message': message
            })
            
            if not is_valid:
                print(f"Skipping {audio_file}: {message}")
                continue
            
            # Извлечение сегмента
            audio_segment = self.extract_segment(str(audio_file))
            if audio_segment is None:
                continue
            
            # Вычисление спектрограммы
            mel_spec = self.compute_melspectrogram(audio_segment)
            
            # Парсинг метаданных
            metadata = self.parse_metadata(str(audio_file))
            metadata['spec_shape'] = mel_spec.shape
            
            # Сохранение промежуточных результатов
            all_specs.append(mel_spec)
            all_metadata.append(metadata)
        
        # Сохранение датасета
        if all_specs:
            # Конвертация в numpy массивы
            specs_array = np.stack(all_specs, axis=0)
            
            # Добавление размерности канала (для CNN)
            specs_array = specs_array[:, np.newaxis, :, :]  # (n_samples, 1, n_mels, time)
            
            # Сохранение спектрограмм
            np.save(output_path / "melspectrograms.npy", specs_array)
            print(f"Saved spectrograms shape: {specs_array.shape}")
            
            # Сохранение метаданных
            metadata_df = pd.DataFrame(all_metadata)
            metadata_df.to_csv(output_path / "metadata.csv", index=False)
            print(f"Saved metadata for {len(metadata_df)} samples")
        
        # Сохранение лога
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(log_file, index=False)
        
        # Сохранение конфигурации
        with open(output_path / "pipeline_config.yaml", 'w') as f:
            yaml.dump({'data_pipeline': self.config}, f)
        
        print(f"Dataset built successfully in {output_dir}")
        return specs_array, metadata_df

# Пример использования
if __name__ == "__main__":
    # Инициализация
    builder = DataBuilder()
    
    # Построение датасета
    # Предполагается структура: input_dir/instrument_name/*.wav
    specs, meta = builder.build_dataset(
        input_dir="raw_audio",
        output_dir="processed_dataset"
    )
    
    # Вывод информации
    print("\n" + "="*50)
    print("ДАТАСЕТ УСПЕШНО СОЗДАН")
    print("="*50)
    print(f"Количество семплов: {len(specs)}")
    print(f"Размерность спектрограмм: {specs.shape}")
    print(f"Колонки метаданных: {list(meta.columns)}")
    print(f"Пример one-hot вектора: {meta.iloc[0]['one_hot_vector'][:10]}...")