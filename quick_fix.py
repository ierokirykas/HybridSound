# quick_test_fixed.py
#!/usr/bin/env python3
"""Быстрый тест исправленной модели"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from src.models.fixed_vae import FixedSizeVAE
from src.data.dataset import AudioDataLoader

def test_model():
    """Тестирование исправленной модели"""
    
    # Создаем тестовую модель
    print("Создание исправленной модели...")
    model = FixedSizeVAE(
        latent_dim=64,
        attribute_dims={
            'instrument_family': 6,
            'instrument_source': 3,
            'brightness': 3,
            'attack_type': 3
        }
    )
    
    # Тестовый вход
    test_input = torch.randn(2, 1, 128, 63)
    print(f"Тестовый вход: {test_input.shape}")
    
    # Прямой проход
    x_recon, mu, log_var, attr_pred = model(test_input)
    print(f"Выход реконструкции: {x_recon.shape}")
    
    if test_input.shape == x_recon.shape:
        print("✓ Размеры совпадают!")
    else:
        print(f"✗ Размеры не совпадают: {test_input.shape} != {x_recon.shape}")
        # Покажем разницу
        print(f"  Высота: {test_input.shape[2]} vs {x_recon.shape[2]}")
        print(f"  Ширина: {test_input.shape[3]} vs {x_recon.shape[3]}")
    
    # Тестирование потерь
    print("\nТестирование функции потерь...")
    test_attributes = {
        'instrument_family': torch.randint(0, 6, (2,)),
        'instrument_source': torch.randint(0, 3, (2,)),
        'brightness': torch.randint(0, 3, (2,)),
        'attack_type': torch.randint(0, 3, (2,))
    }
    
    losses = model.loss_function(test_input, x_recon, mu, log_var, attr_pred, test_attributes)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {losses['recon_loss'].item():.4f}")
    print(f"KL loss: {losses['kl_loss'].item():.4f}")
    
    return model

def test_with_real_data():
    """Тест с реальными данными"""
    
    # Пути к данным
    data_dir = "data/processed/nsynth_processed"
    spectrograms_path = f"{data_dir}/melspectrograms.npy"
    metadata_path = f"{data_dir}/processed_metadata.json"
    
    # Создание DataLoader
    print("\nЗагрузка реальных данных...")
    train_loader, val_loader, attribute_dims = AudioDataLoader.create_dataloaders(
        spectrograms_path=spectrograms_path,
        metadata_path=metadata_path,
        batch_size=4,
        train_ratio=0.8,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Размеры атрибутов: {attribute_dims}")
    
    # Создание модели с правильными размерами атрибутов
    model = FixedSizeVAE(
        latent_dim=64,
        attribute_dims=attribute_dims
    )
    
    # Тестовый батч
    print("\nТестирование на реальных данных...")
    for batch_idx, (x, attributes) in enumerate(train_loader):
        print(f"Батч {batch_idx}:")
        print(f"  Входной размер: {x.shape}")
        
        # Прямой проход
        x_recon, mu, log_var, attr_pred = model(x)
        print(f"  Выходной размер: {x_recon.shape}")
        
        if x.shape == x_recon.shape:
            print("  ✓ Размеры совпадают")
        else:
            print(f"  ✗ Размеры не совпадают")
        
        # Потери
        losses = model.loss_function(x, x_recon, mu, log_var, attr_pred, attributes)
        print(f"  Total loss: {losses['total_loss'].item():.4f}")
        
        if batch_idx >= 1:
            break
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТ ИСПРАВЛЕННОЙ МОДЕЛИ FIXEDSIZEVAE")
    print("=" * 60)
    
    model1 = test_model()
    print("\n" + "=" * 60)
    model2 = test_with_real_data()
    
    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЕН!")
    print("=" * 60)