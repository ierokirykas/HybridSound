# scripts/train_fixed_vae_fixed.py
#!/usr/bin/env python3
"""Обучение исправленной VAE модели с защитой от взрыва KL"""

import sys
from pathlib import Path
import argparse

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Обучение FixedSizeVAE с исправлениями')
    parser.add_argument('--data_dir', type=str, default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--batch_size', type=int, default=32,  # Уменьшено
                       help='Размер батча')
    parser.add_argument('--latent_dim', type=int, default=16,  # Уменьшено
                       help='Размерность латентного пространства')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох обучения')
    parser.add_argument('--lr', type=float, default=5e-4,  # Уменьшено
                       help='Скорость обучения')
    parser.add_argument('--beta', type=float, default=10.0,  # Увеличено
                       help='Вес KL-дивергенции')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/fixed_vae_stable',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для обучения')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Градиентный клиппинг')
    
    args = parser.parse_args()
    
    # Устройство
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA не доступна, используется CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("ОБУЧЕНИЕ FIXEDSIZEVAE (СТАБИЛЬНАЯ ВЕРСИЯ)")
    print("=" * 60)
    print(f"Устройство: {device}")
    print(f"Латентная размерность: {args.latent_dim}")
    print(f"Эпохи: {args.epochs}")
    print(f"Beta (KL weight): {args.beta}")
    print(f"Learning rate: {args.lr}")
    
    # Импорты
    from src.models.fixed_vae import FixedSizeVAE
    from src.data.dataset import AudioDataLoader
    
    # Пути к данным
    data_dir = Path(args.data_dir)
    spectrograms_path = data_dir / "melspectrograms.npy"
    metadata_path = data_dir / "processed_metadata.json"
    
    # Создание DataLoader
    print("\nСоздание DataLoader...")
    train_loader, val_loader, attribute_dims = AudioDataLoader.create_dataloaders(
        spectrograms_path=str(spectrograms_path),
        metadata_path=str(metadata_path),
        batch_size=args.batch_size,
        train_ratio=0.8,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Attribute dims: {attribute_dims}")
    
    # Создание модели
    print("\nСоздание модели...")
    model = FixedSizeVAE(
        latent_dim=args.latent_dim,
        attribute_dims=attribute_dims,
        beta=args.beta,
        attr_weights={
            'instrument_family': 0.5,
            'instrument_source': 0.2,
            'brightness': 0.05,
            'attack_type': 0.01
        }
    )
    
    model.to(device)
    
    # Проверка размеров
    print("Проверка размеров модели...")
    test_input = torch.randn(1, 1, 128, 63).to(device)
    with torch.no_grad():
        x_recon, mu, log_var, _ = model(test_input)
    
    if test_input.shape == x_recon.shape:
        print("✓ Размеры вход/выход совпадают")
        print(f"  mu range: [{mu.min():.3f}, {mu.max():.3f}]")
        print(f"  log_var range: [{log_var.min():.3f}, {log_var.max():.3f}]")
    else:
        print(f"✗ Размеры не совпадают: {test_input.shape} != {x_recon.shape}")
        return
    
    # Оптимизатор и scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # Увеличена регуляризация
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/fixed_vae_stable_{timestamp}")
    
    # Директория для чекпоинтов
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # История обучения
    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'val_recon': [],
        'train_kl': [], 'val_kl': [],
        'train_attr': [], 'val_attr': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # Early stopping patience
    
    # Цикл обучения
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Обучение
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0, 'attr': 0}
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (x, attributes) in enumerate(pbar):
            x = x.to(device)
            attributes = {k: v.to(device) for k, v in attributes.items()}
            
            # Forward pass
            x_recon, mu, log_var, attr_pred = model(x)
            
            # Проверка на NaN
            if torch.isnan(x_recon).any() or torch.isnan(mu).any() or torch.isnan(log_var).any():
                print("Обнаружен NaN в forward pass, пропускаем батч")
                continue
            
            # Loss
            losses = model.loss_function(x, x_recon, mu, log_var, attr_pred, attributes)
            
            # Проверка loss на NaN
            if torch.isnan(losses['total_loss']):
                print("Обнаружен NaN в loss, пропускаем батч")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Статистика
            train_losses['total'] += losses['total_loss'].item()
            train_losses['recon'] += losses['recon_loss'].item()
            train_losses['kl'] += losses['kl_loss'].item()
            train_losses['attr'] += losses['total_attr_loss'].item()
            
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item(),
                'kl': losses['kl_loss'].item()
            })
        
        # Усреднение
        num_batches = len(train_loader)
        for key in train_losses:
            if num_batches > 0:
                train_losses[key] /= num_batches
            else:
                train_losses[key] = 0
        
        # Валидация
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0, 'attr': 0}
        
        with torch.no_grad():
            for x, attributes in val_loader:
                x = x.to(device)
                attributes = {k: v.to(device) for k, v in attributes.items()}
                
                x_recon, mu, log_var, attr_pred = model(x)
                losses = model.loss_function(x, x_recon, mu, log_var, attr_pred, attributes)
                
                val_losses['total'] += losses['total_loss'].item()
                val_losses['recon'] += losses['recon_loss'].item()
                val_losses['kl'] += losses['kl_loss'].item()
                val_losses['attr'] += losses['total_attr_loss'].item()
        
        # Усреднение
        num_batches = len(val_loader)
        for key in val_losses:
            if num_batches > 0:
                val_losses[key] /= num_batches
            else:
                val_losses[key] = 0
        
        # Обновление scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Сохранение в историю
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['train_recon'].append(train_losses['recon'])
        history['val_recon'].append(val_losses['recon'])
        history['train_kl'].append(train_losses['kl'])
        history['val_kl'].append(val_losses['kl'])
        history['train_attr'].append(train_losses['attr'])
        history['val_attr'].append(val_losses['attr'])
        history['learning_rate'].append(current_lr)
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('Recon/train', train_losses['recon'], epoch)
        writer.add_scalar('Recon/val', val_losses['recon'], epoch)
        writer.add_scalar('KL/train', train_losses['kl'], epoch)
        writer.add_scalar('KL/val', val_losses['kl'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Вывод
        print(f"Train Loss: {train_losses['total']:.4f} "
              f"(Recon: {train_losses['recon']:.4f}, "
              f"KL: {train_losses['kl']:.4f}, "
              f"Attr: {train_losses['attr']:.4f})")
        print(f"Val Loss: {val_losses['total']:.4f} "
              f"(Recon: {val_losses['recon']:.4f}, "
              f"KL: {val_losses['kl']:.4f}, "
              f"Attr: {val_losses['attr']:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Early stopping и сохранение лучшей модели
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': {
                    'latent_dim': args.latent_dim,
                    'attribute_dims': attribute_dims,
                    'beta': args.beta
                }
            }
            
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"✓ Новая лучшая модель сохранена (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Ранняя остановка на эпохе {epoch}")
                break
        
        # Регулярное сохранение
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
            print(f"Чекпоинт сохранен: epoch_{epoch}")
    
    # Закрытие TensorBoard
    writer.close()
    
    # Сохранение финальной модели
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'config': {
            'latent_dim': args.latent_dim,
            'attribute_dims': attribute_dims,
            'beta': args.beta
        }
    }
    
    torch.save(final_checkpoint, checkpoint_dir / "final_model.pth")
    
    # Сохранение истории
    with open(checkpoint_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Визуализация
    print("\nСоздание визуализаций...")
    try:
        import matplotlib.pyplot as plt
        
        # График потерь
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history['train_recon'], label='Train')
        axes[0, 1].plot(history['val_recon'], label='Val')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(history['train_kl'], label='Train')
        axes[0, 2].plot(history['val_kl'], label='Val')
        axes[0, 2].set_title('KL Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history['train_attr'], label='Train')
        axes[1, 0].plot(history['val_attr'], label='Val')
        axes[1, 0].set_title('Attribute Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history['learning_rate'], label='LR')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Латентное пространство
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(checkpoint_dir / "training_curves.png", dpi=150)
        plt.close()
        
        # Визуализация реконструкций
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(val_loader))
            x = x[:3].to(device)
            x_recon, mu, log_var, _ = model(x)
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            
            for i in range(3):
                # Оригинал
                axes[i, 0].imshow(x[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                axes[i, 0].set_title(f"Original {i+1}")
                axes[i, 0].set_xlabel("Time")
                axes[i, 0].set_ylabel("Mel-bins")
                
                # Реконструкция
                axes[i, 1].imshow(x_recon[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                axes[i, 1].set_title(f"Reconstructed {i+1}")
                axes[i, 1].set_xlabel("Time")
                axes[i, 1].set_ylabel("Mel-bins")
                
                # Разность
                diff = np.abs(x[i, 0].cpu().numpy() - x_recon[i, 0].cpu().numpy())
                axes[i, 2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
                axes[i, 2].set_title(f"Difference {i+1}")
                axes[i, 2].set_xlabel("Time")
                axes[i, 2].set_ylabel("Mel-bins")
            
            plt.tight_layout()
            plt.savefig(checkpoint_dir / "reconstructions.png", dpi=150)
            plt.close()
        
        print(f"Визуализации сохранены в {checkpoint_dir}")
        
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"Лучшая val loss: {best_val_loss:.4f}")
    print(f"Модель сохранена в: {checkpoint_dir}")
    print(f"Для просмотра графиков запустите: tensorboard --logdir runs/")

if __name__ == "__main__":
    main()