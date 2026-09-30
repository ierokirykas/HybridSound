# scripts/analyze_vae.py
#!/usr/bin/env python3
"""Анализ обученной VAE модели"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import warnings
import librosa
import soundfile as sf
from scipy import signal
from datetime import datetime

warnings.filterwarnings('ignore')

def load_model(checkpoint_path, device='cuda'):
    """Загрузка модели из чекпоинта"""
    print(f"Загрузка модели из {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Загружаем конфигурацию
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 16)
    attribute_dims = config.get('attribute_dims', {})
    
    # Импортируем модель
    from src.models.fixed_vae import FixedSizeVAE
    
    # Создаем модель
    model = FixedSizeVAE(
        latent_dim=latent_dim,
        attribute_dims=attribute_dims,
        beta=config.get('beta', 1.0)
    )
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Модель загружена:")
    print(f"  Латентная размерность: {latent_dim}")
    print(f"  Атрибуты: {list(attribute_dims.keys())}")
    print(f"  Эпоха обучения: {checkpoint.get('epoch', 'unknown')}")
    
    return model, checkpoint

def load_data(data_dir, batch_size=32):
    """Загрузка данных"""
    from src.data.dataset import AudioDataLoader
    
    data_dir = Path(data_dir)
    spectrograms_path = data_dir / "melspectrograms.npy"
    metadata_path = data_dir / "processed_metadata.json"
    
    print(f"\nЗагрузка данных из {data_dir}")
    
    # Создаем DataLoader только для валидации
    _, val_loader, attribute_dims = AudioDataLoader.create_dataloaders(
        spectrograms_path=str(spectrograms_path),
        metadata_path=str(metadata_path),
        batch_size=batch_size,
        train_ratio=0.8,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Загружено {len(val_loader.dataset)} валидационных семплов")
    
    return val_loader, attribute_dims

def compute_reconstruction_metrics(model, dataloader, device='cuda'):
    """Вычисление метрик реконструкции"""
    print("\nВычисление метрик реконструкции...")
    
    model.eval()
    mse_losses = []
    psnr_values = []
    ssim_values = []
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (x, attributes) in enumerate(tqdm(dataloader, desc="Обработка батчей")):
            x = x.to(device)
            
            # Forward pass
            x_recon, mu, log_var, _ = model(x)
            
            # Вычисляем MSE
            mse = F.mse_loss(x_recon, x, reduction='none').mean(dim=(1, 2, 3))
            mse_losses.extend(mse.cpu().numpy())
            
            # Вычисляем PSNR
            for i in range(x.shape[0]):
                # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
                # MAX_I = 1.0 (так как спектрограммы нормализованы в [0, 1])
                mse_val = mse[i].item()
                if mse_val > 0:
                    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse_val)
                else:
                    psnr = 100  # Бесконечность
                psnr_values.append(psnr)
            
            # Сохраняем латентные векторы и метки
            latent_vectors.append(mu.cpu().numpy())
            
            # Сохраняем метки (используем instrument_family)
            if 'instrument_family' in attributes:
                labels.extend(attributes['instrument_family'].cpu().numpy())
    
    # Объединяем латентные векторы
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Вычисляем статистику
    metrics = {
        'mse_mean': float(np.mean(mse_losses)),
        'mse_std': float(np.std(mse_losses)),
        'mse_min': float(np.min(mse_losses)),
        'mse_max': float(np.max(mse_losses)),
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'psnr_min': float(np.min(psnr_values)),
        'psnr_max': float(np.max(psnr_values)),
    }
    
    print(f"✓ Метрики реконструкции:")
    print(f"  MSE: {metrics['mse_mean']:.4f} ± {metrics['mse_std']:.4f}")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} dB ± {metrics['psnr_std']:.2f}")
    
    return metrics, latent_vectors, labels

def visualize_latent_space(latent_vectors, labels, output_dir, label_names=None):
    """Визуализация латентного пространства с помощью t-SNE и PCA"""
    print("\nВизуализация латентного пространства...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Преобразуем метки в строки, если переданы имена
    unique_labels = np.unique(labels)
    if label_names is not None:
        label_strs = [label_names.get(l, str(l)) for l in labels]
    else:
        label_strs = [str(l) for l in labels]
    
    # 1. PCA (2 компоненты)
    print("  Выполнение PCA...")
    pca = PCA(n_components=2, random_state=42)
    latent_pca = pca.fit_transform(latent_vectors)
    
    # 2. t-SNE (2 компоненты)
    print("  Выполнение t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Создаем DataFrame для визуализации
    df_pca = pd.DataFrame({
        'x': latent_pca[:, 0],
        'y': latent_pca[:, 1],
        'label': label_strs,
        'label_id': labels
    })
    
    df_tsne = pd.DataFrame({
        'x': latent_tsne[:, 0],
        'y': latent_tsne[:, 1],
        'label': label_strs,
        'label_id': labels
    })
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA plot
    unique_labels = sorted(df_pca['label'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = df_pca['label'] == label
        axes[0].scatter(df_pca.loc[mask, 'x'], df_pca.loc[mask, 'y'],
                       c=[color], label=label, alpha=0.6, s=20)
    
    axes[0].set_title('PCA проекция латентного пространства', fontsize=14)
    axes[0].set_xlabel('PCA Component 1', fontsize=12)
    axes[0].set_ylabel('PCA Component 2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE plot
    for label, color in zip(unique_labels, colors):
        mask = df_tsne['label'] == label
        axes[1].scatter(df_tsne.loc[mask, 'x'], df_tsne.loc[mask, 'y'],
                       c=[color], label=label, alpha=0.6, s=20)
    
    axes[1].set_title('t-SNE проекция латентного пространства', fontsize=14)
    axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сохраняем данные для дальнейшего анализа
    df_pca.to_csv(output_dir / 'latent_pca.csv', index=False)
    df_tsne.to_csv(output_dir / 'latent_tsne.csv', index=False)
    
    # Вычисляем кластерные метрики
    print("  Анализ кластеризации...")
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    if len(unique_labels) > 1:
        try:
            sil_score = silhouette_score(latent_vectors, labels)
            ch_score = calinski_harabasz_score(latent_vectors, labels)
            
            print(f"  Silhouette Score: {sil_score:.3f}")
            print(f"  Calinski-Harabasz Score: {ch_score:.3f}")
            
            # Сохраняем метрики
            cluster_metrics = {
                'silhouette_score': float(sil_score),
                'calinski_harabasz_score': float(ch_score)
            }
            
            return cluster_metrics
        except:
            print("  Не удалось вычислить метрики кластеризации")
    
    return None

def evaluate_attribute_predictions(model, dataloader, device='cuda'):
    """Оценка точности предсказания атрибутов"""
    print("\nОценка предсказания атрибутов...")
    
    model.eval()
    all_predictions = {}
    all_targets = {}
    attribute_accuracies = {}
    
    with torch.no_grad():
        for batch_idx, (x, attributes) in enumerate(tqdm(dataloader, desc="Предсказание атрибутов")):
            x = x.to(device)
            
            # Forward pass
            _, mu, _, attr_preds = model(x)
            
            # Сохраняем предсказания и цели
            for attr_name, pred in attr_preds.items():
                if attr_name not in all_predictions:
                    all_predictions[attr_name] = []
                    all_targets[attr_name] = []
                
                # Предсказанный класс
                pred_class = torch.argmax(pred, dim=1).cpu().numpy()
                all_predictions[attr_name].extend(pred_class)
                
                # Целевой класс
                if attr_name in attributes:
                    target = attributes[attr_name].cpu().numpy()
                    all_targets[attr_name].extend(target)
    
    # Вычисляем accuracy для каждого атрибута
    for attr_name in all_predictions:
        if attr_name in all_targets and len(all_targets[attr_name]) > 0:
            preds = np.array(all_predictions[attr_name])
            targets = np.array(all_targets[attr_name])
            
            accuracy = np.mean(preds == targets)
            attribute_accuracies[attr_name] = float(accuracy)
            
            print(f"  {attr_name}: accuracy = {accuracy:.3f}")
    
    return attribute_accuracies, all_predictions, all_targets

def visualize_reconstructions(model, dataloader, output_dir, device='cuda', n_examples=5):
    """Визуализация примеров реконструкций"""
    print("\nВизуализация реконструкций...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Берем один батч
        x, attributes = next(iter(dataloader))
        x = x[:n_examples].to(device)
        
        # Реконструкция
        x_recon, _, _, _ = model(x)
        
        # Переводим в numpy
        x_np = x.cpu().numpy()
        x_recon_np = x_recon.cpu().numpy()
        
        # Создаем визуализацию
        fig, axes = plt.subplots(n_examples, 2, figsize=(10, 3 * n_examples))
        
        for i in range(n_examples):
            # Оригинал
            axes[i, 0].imshow(x_np[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 0].set_title(f"Оригинал {i+1}")
            axes[i, 0].set_xlabel("Время")
            axes[i, 0].set_ylabel("Мел-банки")
            axes[i, 0].grid(False)
            
            # Реконструкция
            axes[i, 1].imshow(x_recon_np[i, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title(f"Реконструкция {i+1}")
            axes[i, 1].set_xlabel("Время")
            axes[i, 1].set_ylabel("Мел-банки")
            axes[i, 1].grid(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reconstruction_examples.png', dpi=150)
        plt.close()
        
        print(f"✓ Примеры реконструкций сохранены")

def perform_interpolation(model, dataloader, output_dir, device='cuda'):
    """Выполнение интерполяции между инструментами"""
    print("\nВыполнение интерполяции...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Находим примеры разных инструментов
    # Для простоты берем первые 2 семпла из датасета
    model.eval()
    with torch.no_grad():
        # Получаем два разных семпла
        all_samples = []
        all_attributes = []
        
        for x, attributes in dataloader:
            all_samples.append(x)
            all_attributes.append(attributes)
            if len(all_samples) >= 2:
                break
        
        if len(all_samples) < 2:
            print("  Недостаточно данных для интерполяции")
            return
        
        # Берем два разных семпла
        sample1 = all_samples[0][0:1].to(device)  # Первый семпл из первого батча
        sample2 = all_samples[1][0:1].to(device)  # Первый семпл из второго батча
        
        # Кодируем в латентное пространство
        mu1, log_var1 = model.encode(sample1)
        mu2, log_var2 = model.encode(sample2)
        
        z1 = model.reparameterize(mu1, log_var1)
        z2 = model.reparameterize(mu2, log_var2)
        
        # Линейная интерполяция
        n_steps = 7
        interpolated_specs = []
        
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * z1 + alpha * z2
            recon = model.decode(z)
            interpolated_specs.append(recon.cpu().numpy())
        
        # Визуализация интерполяции
        fig, axes = plt.subplots(n_steps, 1, figsize=(12, 2 * n_steps))
        
        for i, (alpha, spec) in enumerate(zip(np.linspace(0, 1, n_steps), interpolated_specs)):
            axes[i].imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(f"Интерполяция α={alpha:.2f}")
            axes[i].set_xlabel("Время")
            axes[i].set_ylabel("Мел-банки")
            axes[i].grid(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'interpolation_example.png', dpi=150)
        plt.close()
        
        # Сохраняем латентные векторы
        interpolation_data = {
            'z1': z1.cpu().numpy().tolist(),
            'z2': z2.cpu().numpy().tolist(),
            'interpolation_steps': np.linspace(0, 1, n_steps).tolist()
        }
        
        with open(output_dir / 'interpolation_data.json', 'w') as f:
            json.dump(interpolation_data, f, indent=2)
        
        print(f"✓ Интерполяция выполнена и сохранена")

def generate_audio_from_spectrogram(spectrogram, output_path, sr=16000):
    """Генерация аудио из спектрограммы (упрощенный вокодер)"""
    try:
        # Простой Griffin-Lim алгоритм
        # Преобразуем из децибел обратно в амплитуду
        # (предполагаем, что спектрограмма в децибелах)
        mel_spec_db = spectrogram
        
        # Обратное преобразование из децибел
        mel_spec = librosa.db_to_power(mel_spec_db)
        
        # Обратное мел-преобразование (приблизительное)
        # Используем псевдообратное мел-преобразование
        n_fft = 2048
        hop_length = 512
        
        # Создаем случайную фазу
        magnitude = np.sqrt(mel_spec)
        phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=mel_spec.shape))
        
        # Обратное STFT
        stft_matrix = magnitude * phase
        y = librosa.istft(stft_matrix, hop_length=hop_length, win_length=n_fft)
        
        # Нормализация
        y = y / np.max(np.abs(y)) * 0.8
        
        # Сохраняем аудио
        sf.write(output_path, y, sr)
        return True
    except Exception as e:
        print(f"  Ошибка генерации аудио: {e}")
        return False

def create_audio_demo(model, dataloader, output_dir, device='cuda'):
    """Создание демо-аудио файлов"""
    print("\nСоздание демо-аудио файлов...")
    
    audio_dir = Path(output_dir) / "audio_demos"
    audio_dir.mkdir(exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Берем несколько семплов
        samples_to_process = 3
        
        for i in range(samples_to_process):
            # Получаем семпл
            x, _ = next(iter(dataloader))
            x_sample = x[0:1].to(device)
            
            # Реконструкция
            x_recon, _, _, _ = model(x_sample)
            
            # Конвертируем в numpy
            original_spec = x_sample[0, 0].cpu().numpy()
            recon_spec = x_recon[0, 0].cpu().numpy()
            
            # Генерируем аудио
            original_audio_path = audio_dir / f"sample_{i}_original.wav"
            recon_audio_path = audio_dir / f"sample_{i}_reconstructed.wav"
            
            # Для простоты сохраняем спектрограммы как изображения
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.title("Оригинальная спектрограмма")
            plt.xlabel("Время")
            plt.ylabel("Мел-банки")
            
            plt.subplot(1, 2, 2)
            plt.imshow(recon_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.title("Реконструированная спектрограмма")
            plt.xlabel("Время")
            plt.ylabel("Мел-банки")
            
            plt.tight_layout()
            plt.savefig(audio_dir / f"sample_{i}_comparison.png", dpi=150)
            plt.close()
        
        print(f"✓ Демо-файлы сохранены в {audio_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ обученной VAE модели')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту модели')
    parser.add_argument('--data_dir', type=str, 
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Директория для сохранения результатов анализа')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча для анализа')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для вычислений')
    
    args = parser.parse_args()
    
    # Устройство
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA не доступна, используется CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    print("=" * 70)
    print("АНАЛИЗ ОБУЧЕННОЙ VAE МОДЕЛИ")
    print("=" * 70)
    print(f"Чекпоинт: {args.checkpoint}")
    print(f"Данные: {args.data_dir}")
    print(f"Устройство: {device}")
    
    # Создаем директорию для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Загрузка модели
    model, checkpoint = load_model(args.checkpoint, device)
    
    # 2. Загрузка данных
    dataloader, attribute_dims = load_data(args.data_dir, args.batch_size)
    
    # Создаем маппинг меток для визуализации
    label_names = {
        0: 'bass',
        1: 'brass',
        2: 'flute',
        3: 'guitar',
        4: 'keyboard',
        5: 'mallet',
        6: 'organ',
        7: 'reed',
        8: 'string',
        9: 'synth_lead',
        10: 'vocal'
    }
    
    # 3. Вычисление метрик реконструкции
    recon_metrics, latent_vectors, labels = compute_reconstruction_metrics(
        model, dataloader, device
    )
    
    # 4. Визуализация латентного пространства
    cluster_metrics = visualize_latent_space(
        latent_vectors, labels, output_dir / "latent_space", label_names
    )
    
    # 5. Оценка предсказания атрибутов
    attr_accuracies, all_preds, all_targets = evaluate_attribute_predictions(
        model, dataloader, device
    )
    
    # 6. Визуализация реконструкций
    visualize_reconstructions(
        model, dataloader, output_dir / "reconstructions", device
    )
    
    # 7. Интерполяция
    perform_interpolation(
        model, dataloader, output_dir / "interpolation", device
    )
    
    # 8. Демо-аудио
    create_audio_demo(
        model, dataloader, output_dir / "demos", device
    )
    
    # Сохраняем сводный отчет
    print("\n" + "=" * 70)
    print("СВОДНЫЙ ОТЧЕТ")
    print("=" * 70)
    
    summary = {
        'model_info': {
            'checkpoint': args.checkpoint,
            'latent_dim': checkpoint.get('config', {}).get('latent_dim', 'unknown'),
            'beta': checkpoint.get('config', {}).get('beta', 'unknown'),
            'training_epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown')
        },
        'data_info': {
            'dataset': args.data_dir,
            'validation_samples': len(dataloader.dataset),
            'attribute_dimensions': attribute_dims
        },
        'reconstruction_metrics': recon_metrics,
        'attribute_accuracies': attr_accuracies,
        'cluster_metrics': cluster_metrics,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Выводим сводку
    print(f"\nМодель:")
    print(f"  Латентная размерность: {summary['model_info']['latent_dim']}")
    print(f"  Beta (KL weight): {summary['model_info']['beta']}")
    print(f"  Лучшая val loss: {summary['model_info']['best_val_loss']:.4f}")
    
    print(f"\nДанные:")
    print(f"  Валидационных семплов: {summary['data_info']['validation_samples']}")
    
    print(f"\nМетрики реконструкции:")
    print(f"  MSE: {summary['reconstruction_metrics']['mse_mean']:.4f}")
    print(f"  PSNR: {summary['reconstruction_metrics']['psnr_mean']:.2f} dB")
    
    print(f"\nТочность предсказания атрибутов:")
    for attr_name, acc in summary['attribute_accuracies'].items():
        print(f"  {attr_name}: {acc:.3f}")
    
    if cluster_metrics:
        print(f"\nМетрики кластеризации:")
        print(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.3f}")
        print(f"  Calinski-Harabasz Score: {cluster_metrics['calinski_harabasz_score']:.3f}")
    
    # Сохраняем отчет
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Результаты анализа сохранены в: {output_dir}")
    print(f"✓ Сводный отчет: {summary_path}")
    print(f"\nСозданные файлы:")
    for item in output_dir.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(output_dir)}")
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 70)

if __name__ == "__main__":
    main()