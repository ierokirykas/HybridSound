# src/data/augmentations_fixed_final.py
import torch
import numpy as np
import random
from typing import Optional

class AudioAugmentationsFixedFinal:
    """Исправленные аугментации, которые не меняют размер тензора"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # Параметры аугментаций (только те, что не меняют размер)
        self.pitch_shift_range = self.config.get('pitch_shift', [-2, 2])
        self.noise_std = self.config.get('noise_std', 0.01)
        self.spec_augment = self.config.get('spec_augment', True)
        
        # УДАЛЕНО: time_stretch, так как он меняет размер
        
        # Вместо time_stretch используем temporal masking
        self.time_mask_size = self.config.get('time_mask_size', 5)
        
        # Исходный размер спектрограмм (нужен для фиксации)
        self.target_shape = (128, 63)  # [mel, time]
    
    def apply_augmentations(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Применение аугментаций, гарантирующих сохранение размера"""
        if not self.enabled or random.random() > 0.5:  # 50% chance
            return spectrogram
        
        # Клонируем тензор
        aug_spec = spectrogram.clone()
        
        # 1. Pitch shift (без изменения размера)
        if random.random() < 0.7:
            shift = random.randint(self.pitch_shift_range[0], self.pitch_shift_range[1])
            if shift != 0:
                # Сдвиг по частотной оси (mel bands)
                aug_spec = torch.roll(aug_spec, shifts=shift, dims=0)
                
                # Обнуляем края
                if shift > 0:
                    aug_spec[:shift, :] = 0
                else:
                    aug_spec[shift:, :] = 0
        
        # 2. Gaussian noise (без изменения размера)
        if random.random() < 0.7:
            noise = torch.randn_like(aug_spec) * self.noise_std
            aug_spec = aug_spec + noise
            aug_spec = torch.clamp(aug_spec, 0, 1)  # Ограничиваем [0, 1]
        
        # 3. Frequency masking (SpecAugment)
        if random.random() < 0.7 and self.spec_augment:
            freq_mask_size = random.randint(1, aug_spec.shape[0] // 5)
            if freq_mask_size > 0:
                freq_start = random.randint(0, aug_spec.shape[0] - freq_mask_size)
                aug_spec[freq_start:freq_start + freq_mask_size, :] = 0
        
        # 4. Time masking (SpecAugment) - вместо time_stretch
        if random.random() < 0.7 and self.spec_augment:
            time_mask_size = random.randint(1, min(self.time_mask_size, aug_spec.shape[1] // 4))
            if time_mask_size > 0:
                time_start = random.randint(0, aug_spec.shape[1] - time_mask_size)
                aug_spec[:, time_start:time_start + time_mask_size] = 0
        
        # 5. Gain (изменение громкости)
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            aug_spec = aug_spec * gain
            aug_spec = torch.clamp(aug_spec, 0, 1)
        
        return aug_spec

class SafeAugmentedDataset(torch.utils.data.Dataset):
    """Безопасный датасет с аугментациями, гарантирующий одинаковый размер"""
    
    def __init__(self, base_dataset, augmentations):
        self.base_dataset = base_dataset
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Получаем элемент
        if hasattr(self.base_dataset, 'dataset'):  # Если это Subset
            spectrogram, attributes = self.base_dataset.dataset[self.base_dataset.indices[idx]]
        else:
            spectrogram, attributes = self.base_dataset[idx]
        
        # Убедимся, что спектрограмма имеет правильную размерность
        if len(spectrogram.shape) == 3:  # [channels, mel, time]
            spec_to_aug = spectrogram.squeeze(0)  # [mel, time]
        else:
            spec_to_aug = spectrogram  # [mel, time]
        
        # Применяем аугментации
        augmented_spec = self.augmentations.apply_augmentations(spec_to_aug)
        
        # Гарантируем правильный размер
        if augmented_spec.shape != spec_to_aug.shape:
            # Если размер изменился (должно быть очень редко), обрезаем или дополняем
            target_shape = spec_to_aug.shape
            
            if augmented_spec.shape[0] != target_shape[0]:  # mel dimension
                augmented_spec = augmented_spec[:target_shape[0], :]
                if augmented_spec.shape[0] < target_shape[0]:
                    padding = torch.zeros(target_shape[0] - augmented_spec.shape[0], 
                                         augmented_spec.shape[1])
                    augmented_spec = torch.cat([augmented_spec, padding], dim=0)
            
            if augmented_spec.shape[1] != target_shape[1]:  # time dimension
                augmented_spec = augmented_spec[:, :target_shape[1]]
                if augmented_spec.shape[1] < target_shape[1]:
                    padding = torch.zeros(augmented_spec.shape[0], 
                                         target_shape[1] - augmented_spec.shape[1])
                    augmented_spec = torch.cat([augmented_spec, padding], dim=1)
        
        # Возвращаем к исходной размерности
        if len(spectrogram.shape) == 3:
            augmented_spec = augmented_spec.unsqueeze(0)  # [1, mel, time]
        
        return augmented_spec, attributes