# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

class AudioSpectrogramDataset(Dataset):
    """Датасет для спектрограмм и атрибутов"""
    
    def __init__(self, 
                 spectrograms_path: str,
                 metadata_path: str,
                 attribute_names: List[str] = None,
                 transform=None):
        """
        Args:
            spectrograms_path: путь к .npy файлу со спектрограммами
            metadata_path: путь к JSON файлу с метаданными
            attribute_names: список имен атрибутов для использования
            transform: трансформации для спектрограмм
        """
        self.spectrograms = np.load(spectrograms_path)
        self.transform = transform
        
        # Загрузка метаданных
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Если не указаны атрибуты, берем все из метаданных
        if attribute_names is None:
            # Берем атрибуты из первого элемента
            if self.metadata:
                sample_meta = self.metadata[0]
                self.attribute_names = [
                    'instrument_family', 
                    'instrument_source',
                    'brightness',
                    'attack_type'
                ]
            else:
                self.attribute_names = []
        else:
            self.attribute_names = attribute_names
        
        # Создаем маппинг категорий для каждого атрибута
        self.attr_to_idx = self._build_attribute_mappings()
        
        # Проверка согласованности данных
        assert len(self.spectrograms) == len(self.metadata), \
            f"Количество спектрограмм ({len(self.spectrograms)}) не совпадает с метаданными ({len(self.metadata)})"
    
    def _build_attribute_mappings(self) -> Dict[str, Dict[str, int]]:
        """Создание маппинга категорий -> индексы для каждого атрибута"""
        attr_to_idx = {}
        
        for attr_name in self.attribute_names:
            # Собираем уникальные значения атрибута
            unique_values = set()
            for meta in self.metadata:
                if attr_name in meta:
                    unique_values.add(str(meta[attr_name]))
            
            # Создаем маппинг
            sorted_values = sorted(list(unique_values))
            attr_to_idx[attr_name] = {val: idx for idx, val in enumerate(sorted_values)}
        
        return attr_to_idx
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Получение одного элемента датасета"""
        # Спектрограмма
        spec = self.spectrograms[idx]
        
        if self.transform:
            spec = self.transform(spec)
        
        # Конвертируем в тензор
        spec_tensor = torch.FloatTensor(spec)
        
        # Метаданные
        meta = self.metadata[idx]
        
        # Атрибуты
        attributes = {}
        for attr_name in self.attribute_names:
            if attr_name in meta:
                value = str(meta[attr_name])
                idx_tensor = torch.tensor(self.attr_to_idx[attr_name][value])
                attributes[attr_name] = idx_tensor
            else:
                # Если атрибут отсутствует, используем 0 (можно изменить)
                attributes[attr_name] = torch.tensor(0)
        
        return spec_tensor, attributes
    
    def get_attribute_dimensions(self) -> Dict[str, int]:
        """Получение размерности каждого атрибута (количество классов)"""
        return {name: len(mapping) for name, mapping in self.attr_to_idx.items()}
    
    def get_sample_by_attributes(self, **attributes) -> List[int]:
        """Получение индексов семплов по заданным атрибутам"""
        indices = []
        
        for idx, meta in enumerate(self.metadata):
            match = True
            for attr_name, attr_value in attributes.items():
                if attr_name not in meta or str(meta[attr_name]) != str(attr_value):
                    match = False
                    break
            
            if match:
                indices.append(idx)
        
        return indices

class AudioDataLoader:
    """Обертка для создания DataLoader с правильной пакетной обработкой"""
    
    @staticmethod
    def collate_fn(batch):
        """Функция для пакетной обработки"""
        specs = []
        attributes_dict = {}
        
        # Собираем все спектрограммы
        for spec, attrs in batch:
            specs.append(spec)
            
            # Собираем атрибуты по именам
            for attr_name, attr_value in attrs.items():
                if attr_name not in attributes_dict:
                    attributes_dict[attr_name] = []
                attributes_dict[attr_name].append(attr_value)
        
        # Конкатенация спектрограмм
        specs_batch = torch.stack(specs, dim=0)
        
        # Конкатенация атрибутов
        attributes_batch = {}
        for attr_name, values in attributes_dict.items():
            attributes_batch[attr_name] = torch.stack(values, dim=0)
        
        return specs_batch, attributes_batch
    
    @classmethod
    def create_dataloaders(cls,
                          spectrograms_path: str,
                          metadata_path: str,
                          batch_size: int = 32,
                          train_ratio: float = 0.8,
                          shuffle: bool = True,
                          num_workers: int = 2):
        """Создание train и validation DataLoader"""
        
        # Полный датасет
        full_dataset = AudioSpectrogramDataset(
            spectrograms_path=spectrograms_path,
            metadata_path=metadata_path
        )
        
        # Разделение на train/val
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Создание DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=cls.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cls.collate_fn,
            pin_memory=True
        )
        
        return train_loader, val_loader, full_dataset.get_attribute_dimensions()