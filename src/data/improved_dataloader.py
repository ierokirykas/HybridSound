# src/data/improved_dataloader.py
"""
Улучшенный DataLoader с фильтрацией семплов по атаке и нормализацией
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AttackAwareAudioDataset(Dataset):
    """Датасет с учётом атаки инструментов"""
    
    def __init__(self, 
                 spectrograms_path,
                 metadata_path,
                 attack_threshold=0.3,
                 attack_duration_ms=100,
                 offset_duration_ms=50,  # Сдвиг от начала для пропуска атаки
                 target_duration=2.0,
                 sample_rate=16000,
                 transform=None):
        """
        Args:
            spectrograms_path: путь к .npy файлу со спектрограммами
            metadata_path: путь к JSON файлу с метаданными
            attack_threshold: порог для определения силы атаки (0-1)
            attack_duration_ms: длительность сегмента атаки в мс
            offset_duration_ms: сдвиг от начала (пропускаем атаку)
            target_duration: целевая длительность в секундах
            sample_rate: частота дискретизации
            transform: трансформации
        """
        self.spectrograms = np.load(spectrograms_path)
        self.sample_rate = sample_rate
        self.attack_threshold = attack_threshold
        self.attack_duration_ms = attack_duration_ms
        self.offset_duration_ms = offset_duration_ms
        self.target_duration = target_duration
        self.transform = transform
        
        # Загрузка метаданных
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Вычисляем параметры для сегментации
        self.attack_samples = int(attack_duration_ms * sample_rate / 1000)
        self.offset_samples = int(offset_duration_ms * sample_rate / 1000)
        self.target_samples = int(target_duration * sample_rate)
        
        # Анализируем атаку в семплах
        self.attack_strengths = self._analyze_attack_strengths()
        
        # Фильтруем семплы с слишком резкой атакой
        self.filtered_indices = self._filter_by_attack_strength()
        
        print(f"Исходное количество семплов: {len(self.spectrograms)}")
        print(f"После фильтрации по атаке: {len(self.filtered_indices)}")
        print(f"Средняя сила атаки: {np.mean(self.attack_strengths):.3f}")
    
    def _analyze_attack_strengths(self):
        """Анализирует силу атаки в семплах"""
        attack_strengths = []
        
        for i in range(len(self.spectrograms)):
            spec = self.spectrograms[i, 0]  # (n_mels, time)
            
            # Усредняем по частотам для получения временной огибающей
            temporal_envelope = np.mean(spec, axis=0)
            
            # Находим энергию в сегменте атаки
            attack_segment = temporal_envelope[:self.attack_samples]
            sustain_segment = temporal_envelope[self.attack_samples:self.attack_samples*2]
            
            if len(attack_segment) > 0 and len(sustain_segment) > 0:
                attack_energy = np.mean(attack_segment)
                sustain_energy = np.mean(sustain_segment)
                
                if sustain_energy > 0:
                    # Отношение атаки к сустейну
                    strength = attack_energy / sustain_energy
                else:
                    strength = 1.0
                
                # Нормализуем
                strength = np.clip(strength, 0, 5) / 5
            else:
                strength = 0.5
            
            attack_strengths.append(strength)
        
        return np.array(attack_strengths)
    
    def _filter_by_attack_strength(self, max_strength=0.7):
        """Фильтрует семплы с слишком резкой атакой"""
        indices = np.where(self.attack_strengths < max_strength)[0]
        return indices.tolist()
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # Получаем индекс после фильтрации
        original_idx = self.filtered_indices[idx]
        
        # Спектрограмма
        spec = self.spectrograms[original_idx]
        
        # Применяем трансформацию если есть
        if self.transform:
            spec = self.transform(spec)
        
        # Метаданные
        meta = self.metadata[original_idx]
        
        # Добавляем информацию об атаке
        attack_strength = self.attack_strengths[original_idx]
        
        # Конвертируем в тензор
        spec_tensor = torch.FloatTensor(spec)
        
        # Атрибуты
        attributes = {
            'instrument_family': torch.tensor(meta.get('instrument_family_idx', 0)),
            'instrument_source': torch.tensor(meta.get('instrument_source_idx', 0)),
            'attack_strength': torch.tensor(attack_strength, dtype=torch.float32)
        }
        
        return spec_tensor, attributes
    
    def get_attack_statistics(self):
        """Возвращает статистику по атаке"""
        return {
            'mean': float(np.mean(self.attack_strengths)),
            'std': float(np.std(self.attack_strengths)),
            'min': float(np.min(self.attack_strengths)),
            'max': float(np.max(self.attack_strengths)),
            'histogram': np.histogram(self.attack_strengths, bins=10)[0].tolist()
        }

class TemporalAugmentation:
    """Аугментации для улучшения временных характеристик"""
    
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
    
    def time_stretch(self, spec, rate=1.0):
        """Растяжение/сжатие по времени"""
        import torch.nn.functional as F
        
        # rate > 1: ускорение, rate < 1: замедление
        new_length = int(spec.shape[-1] / rate)
        
        # Интерполяция
        spec_reshaped = spec.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        spec_stretched = F.interpolate(spec_reshaped, 
                                      size=(spec.shape[-2], new_length),
                                      mode='bilinear',
                                      align_corners=False)
        
        return spec_stretched.squeeze()
    
    def random_time_crop(self, spec, crop_ratio=0.8):
        """Случайный кроп по времени"""
        time_len = spec.shape[-1]
        crop_len = int(time_len * crop_ratio)
        start = torch.randint(0, time_len - crop_len, (1,)).item()
        
        return spec[:, start:start+crop_len]
    
    def add_temporal_noise(self, spec, noise_level=0.01):
        """Добавление шума во временной области"""
        noise = torch.randn_like(spec) * noise_level
        return spec + noise
    
    def smooth_temporal_transitions(self, spec, kernel_size=5):
        """Сглаживание временных переходов"""
        import torch.nn.functional as F
        
        # 1D усредняющий фильтр по времени
        kernel = torch.ones(1, 1, 1, kernel_size) / kernel_size
        spec_reshaped = spec.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        
        # Применяем свертку
        smoothed = F.conv2d(spec_reshaped, kernel, padding=(0, kernel_size//2))
        
        return smoothed.squeeze()

def create_improved_dataloader(spectrograms_path,
                               metadata_path,
                               batch_size=32,
                               attack_threshold=0.3,
                               offset_ms=50,
                               use_augmentation=True,
                               train_ratio=0.8):
    """Создание улучшенного DataLoader"""
    
    # Создаем датасет с учетом атаки
    dataset = AttackAwareAudioDataset(
        spectrograms_path=spectrograms_path,
        metadata_path=metadata_path,
        attack_threshold=attack_threshold,
        offset_duration_ms=offset_ms
    )
    
    # Разделяем на train/val
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Применяем аугментацию к train
    if use_augmentation:
        train_dataset = AugmentedDataset(train_dataset)
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset

class AugmentedDataset(Dataset):
    """Датасет с аугментациями"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.augmenter = TemporalAugmentation()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        spec, attrs = self.base_dataset[idx]
        
        # Случайные аугментации
        if torch.rand(1).item() > 0.5:
            # Случайное растяжение времени
            rate = 0.8 + torch.rand(1).item() * 0.4  # от 0.8 до 1.2
            spec = self.augmenter.time_stretch(spec, rate)
        
        if torch.rand(1).item() > 0.7:
            # Случайный кроп
            spec = self.augmenter.random_time_crop(spec, crop_ratio=0.9)
        
        if torch.rand(1).item() > 0.8:
            # Добавление шума
            spec = self.augmenter.add_temporal_noise(spec, noise_level=0.005)
        
        # Всегда немного сглаживаем
        spec = self.augmenter.smooth_temporal_transitions(spec, kernel_size=3)
        
        return spec, attrs