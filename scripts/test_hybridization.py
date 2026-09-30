# scripts/test_hybridization.py
#!/usr/bin/env python3
"""Тестирование гибридизации тембров с помощью обученной VAE"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class HybridizationTester:
    def __init__(self, checkpoint_path, data_dir, device='cuda'):
        self.device = torch.device(device)
        
        # Загрузка модели
        print("Загрузка модели...")
        self.model, self.checkpoint = self.load_model(checkpoint_path)
        self.model.eval()
        
        # Загрузка данных
        print("Загрузка данных...")
        self.dataloader, self.metadata = self.load_data(data_dir)
        
        # Подготовка инструментов для гибридизации
        self.instruments = self.prepare_instruments()
        
    def load_model(self, checkpoint_path):
        """Загрузка модели из чекпоинта"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        from src.models.fixed_vae import FixedSizeVAE
        
        model = FixedSizeVAE(
            latent_dim=checkpoint['config']['latent_dim'],
            attribute_dims=checkpoint['config']['attribute_dims'],
            beta=checkpoint['config']['beta']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model, checkpoint
    
    def load_data(self, data_dir):
        """Загрузка данных"""
        from src.data.dataset import AudioDataLoader
        
        data_dir = Path(data_dir)
        spectrograms_path = data_dir / "melspectrograms.npy"
        metadata_path = data_dir / "processed_metadata.json"
        
        # Создаем DataLoader
        _, val_loader, _ = AudioDataLoader.create_dataloaders(
            spectrograms_path=str(spectrograms_path),
            metadata_path=str(metadata_path),
            batch_size=1,  # По одному для точного контроля
            train_ratio=0.8,
            shuffle=False,
            num_workers=0
        )
        
        # Загружаем метаданные
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return val_loader, metadata
    
    def prepare_instruments(self):
        """Подготовка инструментов для гибридизации"""
        print("Подготовка инструментов для гибридизации...")
        
        instruments = {}
        
        # Получаем оригинальный датасет из DataLoader
        # self.dataloader.dataset может быть Subset, нужно получить доступ к оригинальному датасету
        if hasattr(self.dataloader.dataset, 'dataset'):
            # Если это Subset (после random_split)
            original_dataset = self.dataloader.dataset.dataset
            subset_indices = self.dataloader.dataset.indices
        else:
            # Если это оригинальный датасет
            original_dataset = self.dataloader.dataset
            subset_indices = list(range(len(original_dataset)))
        
        # Создаем обратное отображение для индексов инструментов
        # Нужно получить маппинг индексов к именам инструментов
        attr_to_idx = original_dataset.attr_to_idx['instrument_family']
        idx_to_attr = {v: k for k, v in attr_to_idx.items()}
        
        # Собираем индексы по инструментальным семействам в пределах валидационного набора
        for subset_idx, original_idx in enumerate(subset_indices):
            # Получаем семпл из датасета
            x, attributes = self.dataloader.dataset[subset_idx]
            
            # Получаем instrument_family из атрибутов
            family_idx = attributes['instrument_family'].item()
            family_name = idx_to_attr.get(family_idx, f"unknown_{family_idx}")
            
            if family_name not in instruments:
                instruments[family_name] = {
                    'subset_indices': [],  # Индексы в Subset
                    'original_indices': [],  # Оригинальные индексы
                    'samples': [],
                    'latent_vectors': []
                }
            
            instruments[family_name]['subset_indices'].append(subset_idx)
            instruments[family_name]['original_indices'].append(original_idx)
        
        # Теперь получаем латентные векторы для каждого семейства
        for family_name, data in instruments.items():
            print(f"  Обработка {family_name}: {len(data['subset_indices'])} семплов")
            
            # Выбираем до 5 случайных семплов
            if len(data['subset_indices']) > 5:
                selected_subset_idx = np.random.choice(data['subset_indices'], 5, replace=False)
            else:
                selected_subset_idx = data['subset_indices']
            
            for subset_idx in selected_subset_idx:
                # Получаем семпл через DataLoader
                x, _ = self.dataloader.dataset[subset_idx]
                x = x.unsqueeze(0).to(self.device)
                
                # Кодируем в латентное пространство
                with torch.no_grad():
                    mu, _ = self.model.encode(x)
                    data['latent_vectors'].append(mu.cpu().numpy())
                    data['samples'].append(x.cpu().numpy())
            
            # Вычисляем средний латентный вектор для семейства
            if data['latent_vectors']:
                data['avg_latent'] = np.mean(data['latent_vectors'], axis=0)
                print(f"    ✓ Получено {len(data['latent_vectors'])} латентных векторов")
            else:
                data['avg_latent'] = None
                print(f"    ✗ Не удалось получить латентные векторы")
        
        # Удаляем семейства без латентных векторов
        instruments = {k: v for k, v in instruments.items() if v['avg_latent'] is not None}
        
        print(f"\nПодготовлено {len(instruments)} инструментальных семейств:")
        for family, data in instruments.items():
            print(f"  {family}: {len(data['latent_vectors'])} семплов")
        
        return instruments
        
    def linear_interpolation(self, z1, z2, alpha=0.5):
        """Линейная интерполяция между двумя латентными векторами"""
        z = (1 - alpha) * z1 + alpha * z2
        return z
    
    def spherical_interpolation(self, z1, z2, alpha=0.5):
        """Сферическая интерполяция (slerp)"""
        # Нормализуем векторы
        z1_norm = z1 / np.linalg.norm(z1)
        z2_norm = z2 / np.linalg.norm(z2)
        
        # Вычисляем угол между векторами
        dot = np.clip(np.dot(z1_norm.flatten(), z2_norm.flatten()), -1, 1)
        omega = np.arccos(dot)
        
        if omega < 1e-10:
            return z1
        
        # Интерполяция
        interp = (np.sin((1 - alpha) * omega) * z1 + 
                  np.sin(alpha * omega) * z2) / np.sin(omega)
        
        return interp
    
    def generate_hybrid(self, instrument1, instrument2, alpha=0.5, 
                       interpolation='linear'):
        """Генерация гибрида двух инструментов"""
        if instrument1 not in self.instruments or instrument2 not in self.instruments:
            print(f"Один из инструментов не найден: {instrument1}, {instrument2}")
            return None
        
        z1 = self.instruments[instrument1]['avg_latent']
        z2 = self.instruments[instrument2]['avg_latent']
        
        if z1 is None or z2 is None:
            print(f"Нет латентных векторов для инструментов")
            return None
        
        # Выбираем метод интерполяции
        if interpolation == 'linear':
            z_hybrid = self.linear_interpolation(z1, z2, alpha)
        elif interpolation == 'spherical':
            z_hybrid = self.spherical_interpolation(z1, z2, alpha)
        else:
            z_hybrid = self.linear_interpolation(z1, z2, alpha)
        
        # Декодируем гибрид
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z_hybrid).to(self.device)
            spectrogram = self.model.decode(z_tensor)
        
        return spectrogram.cpu().numpy()
    
    def test_hybrid_pairs(self, pairs, output_dir):
        """Тестирование гибридизации для заданных пар инструментов"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for pair_name, (inst1, inst2) in pairs.items():
            print(f"\nТестирование гибридизации: {inst1} + {inst2}")
            
            # Создаем директорию для пары
            pair_dir = output_dir / f"{inst1}_vs_{inst2}"
            pair_dir.mkdir(exist_ok=True)
            
            # Тестируем разные значения alpha
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            hybrid_specs = []
            
            for alpha in alphas:
                print(f"  alpha={alpha:.2f}")
                
                # Генерируем гибрид
                spec = self.generate_hybrid(inst1, inst2, alpha, interpolation='linear')
                
                if spec is not None:
                    hybrid_specs.append(spec)
                    
                    # Сохраняем спектрограмму
                    plt.figure(figsize=(8, 4))
                    plt.imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Гибрид: {inst1} ({1-alpha:.2f}) + {inst2} ({alpha:.2f})")
                    plt.xlabel("Время")
                    plt.ylabel("Мел-банки")
                    plt.colorbar(label='Амплитуда')
                    plt.tight_layout()
                    plt.savefig(pair_dir / f"hybrid_alpha_{alpha:.2f}.png", dpi=150)
                    plt.close()
            
            # Создаем анимацию/сравнение всех гибридов
            if hybrid_specs:
                self.create_hybrid_comparison(inst1, inst2, hybrid_specs, alphas, pair_dir)
            
            results[pair_name] = {
                'instruments': [inst1, inst2],
                'alphas': alphas,
                'specs_shape': [s.shape for s in hybrid_specs] if hybrid_specs else []
            }
        
        # Сохраняем результаты
        with open(output_dir / "hybridization_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nРезультаты сохранены в: {output_dir}")
        return results
    
    def create_hybrid_comparison(self, inst1, inst2, specs, alphas, output_dir):
        """Создание сравнения всех гибридов"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (spec, alpha) in enumerate(zip(specs, alphas)):
            ax = axes[i]
            img = ax.imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"α = {alpha:.2f}\n{inst1 if alpha==0 else inst2 if alpha==1 else 'гибрид'}")
            ax.set_xlabel("Время")
            ax.set_ylabel("Мел-банки")
            plt.colorbar(img, ax=ax)
        
        # Оставшееся место используем для легенды
        axes[-1].axis('off')
        axes[-1].text(0.1, 0.5, 
                     f"Гибридизация:\n{inst1} + {inst2}\n\n"
                     f"α = доля {inst2}\n"
                     f"(1-α) = доля {inst1}\n\n"
                     f"Линейная интерполяция\nв латентном пространстве",
                     fontsize=12, verticalalignment='center')
        
        plt.suptitle(f"Гибридизация тембров: {inst1} ↔ {inst2}", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "hybridization_comparison.png", dpi=150)
        plt.close()
    
    def generate_audio_demo(self, output_dir):
        """Генерация демо-аудио гибридов"""
        print("\nГенерация демо-аудио...")
        
        audio_dir = Path(output_dir) / "audio_demos"
        audio_dir.mkdir(exist_ok=True)
        
        # Интересные пары для демо
        demo_pairs = {
            'bass_string': ('bass', 'string'),
        'brass_synth': ('brass', 'synth_lead'),
        'vocal_mallet': ('vocal', 'mallet'),
        'keyboard_guitar': ('keyboard', 'guitar'),
        'flute_reed': ('flute', 'reed')
        } 
        for pair_name, (inst1, inst2) in demo_pairs.items():
            pair_audio_dir = audio_dir / pair_name
            pair_audio_dir.mkdir(exist_ok=True)
            
            print(f"  {inst1} + {inst2}")
            
            # Генерируем несколько гибридов
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                spec = self.generate_hybrid(inst1, inst2, alpha)
                
                if spec is not None:
                    # Сохраняем спектрограмму
                    plt.figure(figsize=(6, 3))
                    plt.imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"{inst1} ({1-alpha:.1f}) + {inst2} ({alpha:.1f})")
                    plt.tight_layout()
                    plt.savefig(pair_audio_dir / f"spectrogram_alpha_{alpha:.1f}.png", dpi=150)
                    plt.close()
                    
                    # Пытаемся сгенерировать аудио
                    audio_path = pair_audio_dir / f"hybrid_alpha_{alpha:.1f}.wav"
                    success = self.spec_to_audio(spec[0, 0], audio_path)
                    
                    if success:
                        print(f"    ✓ Аудио сохранено: alpha={alpha:.1f}")
        
        print(f"\nДемо-аудио сохранены в: {audio_dir}")
    
    def spec_to_audio(self, spectrogram, output_path, sr=16000):
        """Преобразование спектрограммы в аудио (упрощенное)"""
        try:
            # Обратное преобразование мел-спектрограммы
            n_fft = 2048
            hop_length = 512
            
            # Используем Griffin-Lim для обратного преобразования
            spec_db = spectrogram
            
            # Преобразуем из нормализованного вида обратно
            spec_power = librosa.db_to_power(spec_db * 80 - 80)  # Примерное преобразование
            
            # Обратное мел-преобразование
            mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
            inv_mel_basis = np.linalg.pinv(mel_basis)
            
            spec_mag = np.dot(inv_mel_basis, spec_power)
            
            # Восстанавливаем фазу (случайная фаза для демо)
            phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=spec_mag.shape))
            stft_matrix = spec_mag * phase
            
            # Обратное STFT
            y = librosa.istft(stft_matrix, hop_length=hop_length, win_length=n_fft)
            
            # Нормализация
            y = y / np.max(np.abs(y)) * 0.8
            
            # Сохраняем аудио
            sf.write(output_path, y, sr)
            return True
            
        except Exception as e:
            print(f"    ✗ Ошибка преобразования: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование гибридизации тембров')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту модели')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--output_dir', type=str,
                       default='hybridization_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для вычислений')
    
    args = parser.parse_args()
    
    # Проверяем наличие чекпоинта
    if not Path(args.checkpoint).exists():
        print(f"Ошибка: чекпоинт не найден: {args.checkpoint}")
        return
    
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ГИБРИДИЗАЦИИ ТЕМБРОВ")
    print("=" * 70)
    
    # Создаем тестер
    tester = HybridizationTester(args.checkpoint, args.data_dir, args.device)
    
    # Определяем пары инструментов для тестирования
    # interesting_pairs = {
    #     'bass_string': ('bass', 'string'),
    #     'brass_synth': ('brass', 'synth_lead'),
    #     'vocal_mallet': ('vocal', 'mallet'),
    #     'keyboard_guitar': ('keyboard', 'guitar'),
    #     'flute_reed': ('flute', 'reed')
    # }
    
    interesting_pairs = {
            'flute_vocal':('flute','vocal'),
            'keybostring':('keyboard','string'),
            'keyling':('keyboard','string')
        }
    # Тестируем гибридизацию
    results = tester.test_hybrid_pairs(interesting_pairs, args.output_dir)
    
    # Генерируем демо-аудио
    tester.generate_audio_demo(args.output_dir)
    
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"Результаты сохранены в: {args.output_dir}")
    print(f"\nПротестированные пары:")
    for pair_name, pair_data in results.items():
        inst1, inst2 = pair_data['instruments']
        print(f"  {pair_name}: {inst1} ↔ {inst2}")

if __name__ == "__main__":
    main()