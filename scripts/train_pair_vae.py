# scripts/train_pair_vae.py
#!/usr/bin/env python3
"""
Обучение VAE модели на паре инструментов.
Специализированная модель для создания гибридов двух конкретных инструментов.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import librosa

warnings.filterwarnings('ignore')

class PairVAE(nn.Module):
    """Специализированная VAE для пары инструментов"""
    
    def __init__(self, latent_dim=32, beta=1.0, input_shape=(1, 128, 63)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_shape = input_shape
        
        # Упрощенная архитектура для пары инструментов
        # Энкодер
        self.encoder = nn.Sequential(
            # Блок 1
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Блок 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Блок 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Flatten()
        )
        
        # Размер после энкодера
        with torch.no_grad():
            test_input = torch.randn(1, *input_shape)
            encoder_output = self.encoder(test_input)
            encoder_output_size = encoder_output.shape[1]
            # Вычисляем форму для unflatten
            self.encoder_output_shape = self._get_encoder_output_shape()
        
        # Линейные слои для параметров распределения
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_var = nn.Linear(encoder_output_size, latent_dim)
        
        # Декодер
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)
        
        # Вычисляем размер после flatten для декодера
        self.decoder_output_size = encoder_output_size
        
        self.decoder = nn.Sequential(
            # Исправляем размеры unflatten на основе вычисленной формы
            nn.Unflatten(1, self.encoder_output_shape),
            
            # Блок 1
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Блок 2
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Блок 3 (специально для получения правильного размера)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Финальный слой
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Инициализация весов
        self._initialize_weights()
    
    def _get_encoder_output_shape(self):
        """Вычисляет форму выходного тензора энкодера до flatten"""
        with torch.no_grad():
            test_input = torch.randn(1, *self.input_shape)
            # Проходим через сверточные слои без flatten
            x = test_input
            x = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)(x)
            x = nn.BatchNorm2d(32)(x)
            x = nn.ReLU()(x)
            x = nn.Dropout2d(0.2)(x)
            
            x = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)(x)
            x = nn.BatchNorm2d(64)(x)
            x = nn.ReLU()(x)
            x = nn.Dropout2d(0.2)(x)
            
            x = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)(x)
            x = nn.BatchNorm2d(128)(x)
            x = nn.ReLU()(x)
            x = nn.Dropout2d(0.2)(x)
            
            return x.shape[1:]  # Возвращаем (C, H, W)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
    
    def loss_function(self, x, reconstruction, mu, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum') / x.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

class PairAudioDataset:
    """Датасет для пары инструментов"""
    
    def __init__(self, data_dir, instrument1, instrument2):
        self.data_dir = Path(data_dir)
        
        # Загружаем спектрограммы
        spectrograms_path = self.data_dir / "melspectrograms.npy"
        self.spectrograms = np.load(spectrograms_path)
        
        # Загружаем метаданные
        metadata_path = self.data_dir / "processed_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Фильтруем индексы по выбранным инструментам
        self.indices = []
        self.labels = []  # 0 для instrument1, 1 для instrument2
        
        for idx, item in enumerate(self.metadata):
            instrument_family = item.get('instrument_family', 'unknown')
            if instrument_family == instrument1:
                self.indices.append(idx)
                self.labels.append(0)
            elif instrument_family == instrument2:
                self.indices.append(idx)
                self.labels.append(1)
        
        print(f"Найдено {len(self.indices)} семплов:")
        print(f"  {instrument1}: {self.labels.count(0)} семплов")
        print(f"  {instrument2}: {self.labels.count(1)} семплов")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        spec = self.spectrograms[actual_idx]
        label = self.labels[idx]
        
        # Конвертируем в тензор
        spec_tensor = torch.FloatTensor(spec)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return spec_tensor, label_tensor

def compute_interpolation_smoothness(model, z1, z2, device='cuda'):
    """Вычисление плавности интерполяции между двумя инструментами"""
    model.eval()
    
    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)
    
    # Линейная интерполяция в латентном пространстве
    z_samples = []
    with torch.no_grad():
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            z_samples.append(z)
        
        # Декодируем все точки
        z_tensor = torch.stack(z_samples).to(device)
        reconstructions = model.decode(z_tensor)
    
    # Вычисляем плавность (среднее изменение между соседними реконструкциями)
    smoothness_scores = []
    for i in range(n_steps - 1):
        diff = torch.mean(torch.abs(reconstructions[i] - reconstructions[i+1])).item()
        smoothness_scores.append(diff)
    
    avg_smoothness = np.mean(smoothness_scores)
    
    return {
        'smoothness_score': float(avg_smoothness),
        'smoothness_std': float(np.std(smoothness_scores))
    }

def evaluate_model(model, test_loader, device='cuda'):
    """Оценка модели на тестовых данных"""
    model.eval()
    
    recon_losses = []
    kl_losses = []
    psnr_values = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            
            reconstruction, mu, log_var = model(x)
            
            # Вычисляем потери
            losses = model.loss_function(x, reconstruction, mu, log_var)
            recon_losses.append(losses['recon_loss'].item())
            kl_losses.append(losses['kl_loss'].item())
            
            # Вычисляем PSNR
            mse = F.mse_loss(reconstruction, x, reduction='mean').item()
            if mse > 0:
                psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
                psnr_values.append(psnr)
    
    metrics = {
        'recon_loss_mean': float(np.mean(recon_losses)),
        'recon_loss_std': float(np.std(recon_losses)),
        'kl_loss_mean': float(np.mean(kl_losses)),
        'kl_loss_std': float(np.std(kl_losses)),
        'psnr_mean': float(np.mean(psnr_values)) if psnr_values else 0,
        'psnr_std': float(np.std(psnr_values)) if psnr_values else 0
    }
    
    return metrics

def train_pair_vae(config):
    """Обучение VAE на паре инструментов"""
    
    # Параметры из конфига
    data_dir = config['data_dir']
    instrument1 = config['instrument1']
    instrument2 = config['instrument2']
    latent_dim = config['latent_dim']
    beta = config['beta']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    device = config['device']
    output_dir = Path(config['output_dir'])
    
    # Создаем директорию для результатов
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"ОБУЧЕНИЕ VAE ДЛЯ ПАРЫ ИНСТРУМЕНТОВ")
    print("=" * 60)
    print(f"Инструмент 1: {instrument1}")
    print(f"Инструмент 2: {instrument2}")
    print(f"Латентность: {latent_dim}")
    print(f"Beta: {beta}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Эпохи: {epochs}")
    print(f"Устройство: {device}")
    
    # Создаем датасет
    print("\nСоздание датасета...")
    dataset = PairAudioDataset(data_dir, instrument1, instrument2)
    
    if len(dataset) < 50:
        print(f"⚠️  Мало данных: только {len(dataset)} семплов")
    
    # Разделение на train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Создаем модель
    print("\nСоздание модели...")
    model = PairVAE(latent_dim=latent_dim, beta=beta)
    model.to(device)
    
    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # История обучения
    history = {
        'train_loss': [], 'test_loss': [],
        'train_recon': [], 'test_recon': [],
        'train_kl': [], 'test_kl': []
    }
    
    best_test_loss = float('inf')
    best_model_state = None
    
    # Обучение
    print("\nНачало обучения...")
    for epoch in range(1, epochs + 1):
        # Тренировка
        model.train()
        train_losses = {'total': [], 'recon': [], 'kl': []}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            
            optimizer.zero_grad()
            reconstruction, mu, log_var = model(x)
            losses = model.loss_function(x, reconstruction, mu, log_var)
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses['total'].append(losses['total_loss'].item())
            train_losses['recon'].append(losses['recon_loss'].item())
            train_losses['kl'].append(losses['kl_loss'].item())
            
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'recon': losses['recon_loss'].item()
            })
        
        # Валидация
        model.eval()
        test_losses = {'total': [], 'recon': [], 'kl': []}
        
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                reconstruction, mu, log_var = model(x)
                losses = model.loss_function(x, reconstruction, mu, log_var)
                
                test_losses['total'].append(losses['total_loss'].item())
                test_losses['recon'].append(losses['recon_loss'].item())
                test_losses['kl'].append(losses['kl_loss'].item())
        
        # Усредняем потери
        avg_train_loss = np.mean(train_losses['total'])
        avg_test_loss = np.mean(test_losses['total'])
        
        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_recon'].append(np.mean(train_losses['recon']))
        history['test_recon'].append(np.mean(test_losses['recon']))
        history['train_kl'].append(np.mean(train_losses['kl']))
        history['test_kl'].append(np.mean(test_losses['kl']))
        
        # Обновление scheduler
        scheduler.step(avg_test_loss)
        
        # Сохранение лучшей модели
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            
            # Сохраняем чекпоинт
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_loss': best_test_loss,
                'history': history,
                'config': config
            }
            
            torch.save(checkpoint, output_dir / "best_model.pth")
        
        # Вывод прогресса
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                  f"Test Loss = {avg_test_loss:.4f}")
    
    # Загружаем лучшую модель для финальной оценки
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Финальная оценка
    print("\nФинальная оценка модели...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Вычисляем плавность интерполяции
    print("Вычисление плавности интерполяции...")
    
    # Получаем средние латентные векторы для каждого инструмента
    model.eval()
    with torch.no_grad():
        # Собираем латентные векторы для каждого инструмента
        z_instrument1 = []
        z_instrument2 = []
        
        for x, labels in test_loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            
            for i, label in enumerate(labels):
                if label == 0:  # instrument1
                    z_instrument1.append(mu[i].cpu().numpy())
                else:  # instrument2
                    z_instrument2.append(mu[i].cpu().numpy())
        
        if z_instrument1 and z_instrument2:
            z1_avg = np.mean(z_instrument1, axis=0)
            z2_avg = np.mean(z_instrument2, axis=0)
            
            z1_tensor = torch.FloatTensor(z1_avg).to(device)
            z2_tensor = torch.FloatTensor(z2_avg).to(device)
            
            smoothness = compute_interpolation_smoothness(model, z1_tensor, z2_tensor, device)
            test_metrics.update(smoothness)
    
    # Сохраняем результаты
    results = {
        'config': config,
        'best_test_loss': float(best_test_loss),
        'test_metrics': test_metrics,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'dataset_info': {
            'total_samples': len(dataset),
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'instrument1_count': dataset.labels.count(0),
            'instrument2_count': dataset.labels.count(1)
        }
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Создаем визуализации
    create_visualizations(model, test_loader, history, output_dir, device)
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"Лучшая test loss: {best_test_loss:.4f}")
    print(f"PSNR: {test_metrics['psnr_mean']:.2f} dB")
    if 'smoothness_score' in test_metrics:
        print(f"Плавность интерполяции: {test_metrics['smoothness_score']:.4f}")
    print(f"Результаты сохранены в: {output_dir}")
    
    return results

def create_visualizations(model, test_loader, history, output_dir, device):
    """Создание визуализаций для анализа модели"""
    try:
        import matplotlib.pyplot as plt
        
        # 1. Графики обучения
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['test_loss'], label='Test')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['train_recon'], label='Train')
        axes[1].plot(history['test_recon'], label='Test')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(history['train_kl'], label='Train')
        axes[2].plot(history['test_kl'], label='Test')
        axes[2].set_title('KL Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()
        
        # 2. Примеры реконструкций
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x[:3].to(device)
            reconstruction, _, _ = model(x)
            
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            
            for i in range(3):
                axes[i, 0].imshow(x[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                axes[i, 0].set_title(f"Original {i+1}")
                axes[i, 0].set_xlabel("Time")
                axes[i, 0].set_ylabel("Mel-bins")
                
                axes[i, 1].imshow(reconstruction[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
                axes[i, 1].set_title(f"Reconstructed {i+1}")
                axes[i, 1].set_xlabel("Time")
                axes[i, 1].set_ylabel("Mel-bins")
            
            plt.tight_layout()
            plt.savefig(output_dir / "reconstructions.png", dpi=150)
            plt.close()
        
        # 3. Интерполяция между инструментами
        if 'config' in history:
            # Находим средние латентные векторы для каждого инструмента
            instrument1 = history['config']['instrument1']
            instrument2 = history['config']['instrument2']
            
            # Создаем интерполяцию
            n_steps = 5
            alphas = np.linspace(0, 1, n_steps)
            
            with torch.no_grad():
                # Получаем по одному примеру каждого инструмента
                examples_inst1 = []
                examples_inst2 = []
                
                for x, labels in test_loader:
                    for i, label in enumerate(labels):
                        if label == 0 and len(examples_inst1) < 1:
                            examples_inst1.append(x[i:i+1])
                        elif label == 1 and len(examples_inst2) < 1:
                            examples_inst2.append(x[i:i+1])
                    
                    if len(examples_inst1) >= 1 and len(examples_inst2) >= 1:
                        break
                
                if examples_inst1 and examples_inst2:
                    x1 = examples_inst1[0].to(device)
                    x2 = examples_inst2[0].to(device)
                    
                    mu1, _ = model.encode(x1)
                    mu2, _ = model.encode(x2)
                    
                    # Интерполяция
                    interpolated_specs = []
                    for alpha in alphas:
                        z = (1 - alpha) * mu1 + alpha * mu2
                        recon = model.decode(z)
                        interpolated_specs.append(recon.cpu().numpy())
                    
                    # Визуализация
                    fig, axes = plt.subplots(n_steps, 1, figsize=(10, 2 * n_steps))
                    
                    for i, (alpha, spec) in enumerate(zip(alphas, interpolated_specs)):
                        axes[i].imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
                        axes[i].set_title(f"Интерполяция: {instrument1} ({1-alpha:.2f}) + {instrument2} ({alpha:.2f})")
                        axes[i].set_xlabel("Время")
                        axes[i].set_ylabel("Мел-банки")
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "interpolation_example.png", dpi=150)
                    plt.close()
        
        print(f"✓ Визуализации сохранены")
        
    except Exception as e:
        print(f"⚠️  Ошибка при создании визуализаций: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение VAE на паре инструментов')
    parser.add_argument('--config', type=str, required=True,
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для обучения (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Основные параметры
    pair_config = config['hyperparam_pair_search']
    
    # Создаем конфиг для обучения
    train_config = {
        'data_dir': args.data_dir,
        'instrument1': pair_config['instrument_pair']['instrument1'],
        'instrument2': pair_config['instrument_pair']['instrument2'],
        'latent_dim': None,  # Будет установлено из сетки
        'beta': None,
        'learning_rate': None,
        'batch_size': None,
        'epochs': pair_config['training']['epochs'],
        'device': args.device,
        'output_dir': None  # Будет установлено для каждого эксперимента
    }
    
    # Запускаем обучение (в реальном использовании здесь будет цикл по параметрам)
    # Для демонстрации используем первые значения из сетки
    param_grid = pair_config['param_grid']
    
    print("Для полного гиперпараметрического поиска используйте hyperparameter_search_pair.py")
    print("Этот скрипт предназначен для одного эксперимента.")
    
    # Запрашиваем параметры у пользователя
    print("\nВыберите параметры:")
    print(f"Доступные latent_dim: {param_grid['latent_dim']}")
    latent_dim = int(input(f"latent_dim (выберите одно): "))
    
    print(f"\nДоступные beta: {param_grid['beta']}")
    beta = float(input(f"beta (выберите одно): "))
    
    print(f"\nДоступные learning_rate: {param_grid['learning_rate']}")
    learning_rate = float(input(f"learning_rate (выберите одно): "))
    
    print(f"\nДоступные batch_size: {param_grid['batch_size']}")
    batch_size = int(input(f"batch_size (выберите одно): "))
    
    # Создаем директорию для результатов
    output_dir = Path(pair_config['results_dir']) / \
                f"{train_config['instrument1']}_{train_config['instrument2']}" / \
                f"latent{latent_dim}_beta{beta}_lr{learning_rate}_bs{batch_size}"
    
    # Обновляем конфиг
    train_config.update({
        'latent_dim': latent_dim,
        'beta': beta,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'output_dir': str(output_dir)
    })
    
    # Запускаем обучение
    train_pair_vae(train_config)

if __name__ == "__main__":
    main()