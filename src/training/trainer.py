# src/training/trainer.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class VAETrainer:
    """Класс для обучения VAE модели"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device: str = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 max_grad_norm: float = 1.0):
        """
        Args:
            model: модель VAE
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            device: устройство для обучения (cuda/cpu)
            learning_rate: скорость обучения
            weight_decay: вес для L2 регуляризации
            max_grad_norm: максимальная норма градиента для clipping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_grad_norm = max_grad_norm
        
        # Устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # TensorBoard writer
        self.writer = None
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'train_attr_loss': [],
            'val_attr_loss': []
        }
    
    def setup_tensorboard(self, log_dir: str = None):
        """Настройка TensorBoard для визуализации"""
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"runs/vae_experiment_{timestamp}"
        
        self.writer = SummaryWriter(log_dir)
        return log_dir
    
    def train_epoch(self) -> Dict[str, float]:
        """Одна эпоха обучения"""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0,
            'attr': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, attributes) in enumerate(pbar):
            # Перемещаем данные на устройство
            x = x.to(self.device)
            attributes = {k: v.to(self.device) for k, v in attributes.items()}
            
            # Forward pass
            x_recon, mu, log_var, attr_predictions = self.model(x)
            
            # Вычисление потерь
            losses = self.model.loss_function(x, x_recon, mu, log_var, 
                                             attr_predictions, attributes)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            
            self.optimizer.step()
            
            # Сбор статистики
            epoch_losses['total'] += losses['total_loss'].item()
            epoch_losses['recon'] += losses['recon_loss'].item()
            epoch_losses['kl'] += losses['kl_loss'].item()
            epoch_losses['attr'] += losses['total_attr_loss'].item()
            
            # Обновление прогресс-бара
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item(),
                'kl': losses['kl_loss'].item()
            })
        
        # Усреднение по батчам
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'recon': 0.0,
            'kl': 0.0,
            'attr': 0.0
        }
        
        with torch.no_grad():
            for x, attributes in tqdm(self.val_loader, desc="Validation"):
                x = x.to(self.device)
                attributes = {k: v.to(self.device) for k, v in attributes.items()}
                
                x_recon, mu, log_var, attr_predictions = self.model(x)
                
                losses = self.model.loss_function(x, x_recon, mu, log_var,
                                                 attr_predictions, attributes)
                
                val_losses['total'] += losses['total_loss'].item()
                val_losses['recon'] += losses['recon_loss'].item()
                val_losses['kl'] += losses['kl_loss'].item()
                val_losses['attr'] += losses['total_attr_loss'].item()
        
        # Усреднение по батчам
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, 
              num_epochs: int = 100,
              save_dir: str = "checkpoints",
              save_every: int = 10,
              log_to_tensorboard: bool = True,
              early_stopping_patience: int = 20):
        """Основной цикл обучения"""
        
        # Создание директорий
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Настройка TensorBoard
        if log_to_tensorboard and self.writer is None:
            self.setup_tensorboard()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Обучение
            train_losses = self.train_epoch()
            
            # Валидация
            val_losses = self.validate()
            
            # Обновление scheduler
            self.scheduler.step(val_losses['total'])
            
            # Сохранение в историю
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_recon_loss'].append(train_losses['recon'])
            self.history['val_recon_loss'].append(val_losses['recon'])
            self.history['train_kl_loss'].append(train_losses['kl'])
            self.history['val_kl_loss'].append(val_losses['kl'])
            self.history['train_attr_loss'].append(train_losses['attr'])
            self.history['val_attr_loss'].append(val_losses['attr'])
            
            # Логирование в TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_losses['total'], epoch)
                self.writer.add_scalar('Loss/val', val_losses['total'], epoch)
                self.writer.add_scalar('Recon_Loss/train', train_losses['recon'], epoch)
                self.writer.add_scalar('Recon_Loss/val', val_losses['recon'], epoch)
                self.writer.add_scalar('KL_Loss/train', train_losses['kl'], epoch)
                self.writer.add_scalar('KL_Loss/val', val_losses['kl'], epoch)
                self.writer.add_scalar('Attr_Loss/train', train_losses['attr'], epoch)
                self.writer.add_scalar('Attr_Loss/val', val_losses['attr'], epoch)
                
                # Learning rate
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Вывод статистики
            print(f"Train Loss: {train_losses['total']:.4f} "
                  f"(Recon: {train_losses['recon']:.4f}, "
                  f"KL: {train_losses['kl']:.4f}, "
                  f"Attr: {train_losses['attr']:.4f})")
            print(f"Val Loss: {val_losses['total']:.4f} "
                  f"(Recon: {val_losses['recon']:.4f}, "
                  f"KL: {val_losses['kl']:.4f}, "
                  f"Attr: {val_losses['attr']:.4f})")
            
            # Сохранение лучшей модели
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                
                # Сохраняем лучшую модель
                best_model_path = save_path / "best_model.pth"
                self.save_checkpoint(best_model_path, epoch, is_best=True)
                print(f"✓ Новая лучшая модель сохранена: {best_model_path}")
            else:
                patience_counter += 1
            
            # Регулярное сохранение чекпоинтов
            if epoch % save_every == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(checkpoint_path, epoch)
                print(f"Чекпоинт сохранен: {checkpoint_path}")
            
            # Ранняя остановка
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Ранняя остановка на эпохе {epoch}")
                print(f"Val loss не улучшался {early_stopping_patience} эпох")
                break
        
        # Закрытие TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        # Сохранение финальной модели
        final_model_path = save_path / "final_model.pth"
        self.save_checkpoint(final_model_path, num_epochs)
        print(f"\nФинальная модель сохранена: {final_model_path}")
        
        # Сохранение истории обучения
        history_path = save_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"История обучения сохранена: {history_path}")
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf'),
            'history': self.history,
            'model_config': {
                'input_shape': self.model.input_shape,
                'latent_dim': self.model.latent_dim,
                'attribute_dims': self.model.attribute_dims,
                'beta': self.model.beta,
                'attr_weights': self.model.attr_weights
            }
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Загрузка чекпоинта модели"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['history']
    
    def visualize_reconstructions(self, num_samples: int = 5, save_path: str = None):
        """Визуализация реконструкций"""
        import matplotlib.pyplot as plt
        
        self.model.eval()
        
        # Берем несколько семплов из валидационного набора
        x, _ = next(iter(self.val_loader))
        x = x[:num_samples].to(self.device)
        
        with torch.no_grad():
            x_recon, _, _, _ = self.model(x)
        
        # Перемещаем на CPU
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        
        # Создаем визуализацию
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Исходная спектрограмма
            axes[i, 0].imshow(x[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].set_xlabel("Time")
            axes[i, 0].set_ylabel("Mel-bins")
            
            # Реконструированная спектрограмма
            axes[i, 1].imshow(x_recon[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title(f"Reconstructed {i+1}")
            axes[i, 1].set_xlabel("Time")
            axes[i, 1].set_ylabel("Mel-bins")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Визуализация сохранена: {save_path}")
        
        plt.show()
        return fig
    
    def visualize_latent_space(self, save_path: str = None):
        """Визуализация латентного пространства с помощью t-SNE/UMAP"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.model.eval()
            
            # Собираем латентные векторы и метки
            latents = []
            labels = []
            families = []
            
            with torch.no_grad():
                for x, attributes in self.val_loader:
                    x = x.to(self.device)
                    
                    mu, _ = self.model.encode(x)
                    latents.append(mu.cpu().numpy())
                    
                    # Используем instrument_family как метку
                    if 'instrument_family' in attributes:
                        labels.extend(attributes['instrument_family'].numpy())
                    
                    # Сохраняем названия семейств
                    families.extend([str(f) for f in attributes.get('instrument_family', [0] * x.size(0))])
            
            # Конкатенация всех латентных векторов
            latents = np.concatenate(latents, axis=0)
            
            # Применение t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latents_2d = tsne.fit_transform(latents)
            
            # Визуализация
            plt.figure(figsize=(12, 10))
            
            # Уникальные семейства
            unique_families = sorted(set(families))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_families)))
            
            for family, color in zip(unique_families, colors):
                mask = [f == family for f in families]
                if any(mask):
                    plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                               c=[color], label=family, alpha=0.7, s=50)
            
            plt.title("t-SNE визуализация латентного пространства")
            plt.xlabel("t-SNE компонента 1")
            plt.ylabel("t-SNE компонента 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Визуализация латентного пространства сохранена: {save_path}")
            
            plt.show()
            return latents_2d, families
            
        except ImportError:
            print("Для визуализации требуется установить scikit-learn и seaborn")
            return None, None