# scripts/train_final_model.py
#!/usr/bin/env python3
"""Обучение финальной VAE модели с лучшими параметрами"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Обучение финальной VAE модели')
    parser.add_argument('--data_dir', type=str, 
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints/final_vae',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для обучения')
    
    args = parser.parse_args()
    
    # Лучшие параметры из гиперпараметрического поиска
    best_params = {
        'latent_dim': 64,
        'beta': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_family': 0.5,
        'epochs': 150  # Увеличиваем количество эпох
    }
    
    print("=" * 70)
    print("ОБУЧЕНИЕ ФИНАЛЬНОЙ VAE МОДЕЛИ")
    print("=" * 70)
    print("Лучшие параметры из гиперпараметрического поиска:")
    print(f"  latent_dim: {best_params['latent_dim']}")
    print(f"  beta: {best_params['beta']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  weight_family: {best_params['weight_family']}")
    print(f"  epochs: {best_params['epochs']}")
    
    # Команда для запуска обучения
    cmd = [
        'python', 'scripts/train_vae.py',
        '--data_dir', args.data_dir,
        '--batch_size', str(best_params['batch_size']),
        '--latent_dim', str(best_params['latent_dim']),
        '--epochs', str(best_params['epochs']),
        '--lr', str(best_params['learning_rate']),
        '--beta', str(best_params['beta']),
        '--checkpoint_dir', args.checkpoint_dir,
        '--device', args.device,
        '--grad_clip', '1.0'
    ]
    
    print(f"\nКоманда запуска:")
    print(' '.join(cmd))
    
    # Запускаем обучение
    import subprocess
    process = subprocess.Popen(cmd, text=True, encoding='utf-8')
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "=" * 70)
        print("ФИНАЛЬНАЯ МОДЕЛЬ ОБУЧЕНА!")
        print("=" * 70)
        
        # Проверяем результаты
        checkpoint_path = Path(args.checkpoint_dir) / "best_model.pth"
        if checkpoint_path.exists():
            import json
            print(f"Модель сохранена в: {checkpoint_path}")
            
            # Читаем историю обучения
            history_path = Path(args.checkpoint_dir) / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                if history and 'val_loss' in history:
                    best_val_loss = min(history['val_loss'])
                    print(f"Лучшая val loss: {best_val_loss:.4f}")
    else:
        print(f"\nОшибка при обучении (код: {process.returncode})")

if __name__ == "__main__":
    main()