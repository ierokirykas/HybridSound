# scripts/interactive_hybridizer.py
#!/usr/bin/env python3
"""
Интерактивный инструмент для создания гибридов инструментов.
Пользователь выбирает инструменты по номерам и параметры смешивания.
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import librosa
import soundfile as sf
from datetime import datetime
import shutil

warnings.filterwarnings('ignore')

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class InteractiveHybridizer:
    def __init__(self, checkpoint_path, data_dir, device='cuda'):
        self.device = torch.device(device)
        
        print("=" * 60)
        print("ИНТЕРАКТИВНЫЙ ГИБРИДИЗАТОР ТЕМБРОВ")
        print("=" * 60)
        
        # Загрузка модели
        print("\n[1/4] Загрузка модели...")
        self.model, self.checkpoint = self.load_model(checkpoint_path)
        self.model.eval()
        
        # Загрузка данных и подготовка инструментов
        print("[2/4] Загрузка данных...")
        self.data_dir = Path(data_dir)
        self.instruments = self.prepare_instruments_list()
        
        # Инициализация интерфейса
        self.current_selection = None
        self.output_base_dir = Path("hybridization_results/interactive")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        print("[3/4] Подготовка латентных представлений...")
        self.latent_vectors = self.prepare_latent_vectors()
        
        print("[4/4] Готово!")
        print("-" * 60)
    
    def load_model(self, checkpoint_path):
        """Загрузка модели из чекпоинта"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        from src.models.fixed_vae import FixedSizeVAE
        
        model = FixedSizeVAE(
            latent_dim=checkpoint['config']['latent_dim'],
            attribute_dims=checkpoint['config']['attribute_dims'],
            beta=checkpoint['config']['beta']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Информация о модели
        print(f"  Модель: VAE с латентностью {checkpoint['config']['latent_dim']}")
        print(f"  Beta: {checkpoint['config']['beta']}")
        print(f"  Эпоха обучения: {checkpoint.get('epoch', 'unknown')}")
        
        return model, checkpoint
    
    def prepare_instruments_list(self):
        """Подготовка списка инструментов из метаданных"""
        metadata_path = self.data_dir / "processed_metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Собираем уникальные инструментальные семейства
        instruments_set = set()
        for item in metadata:
            family = item.get('instrument_family', 'unknown')
            if family and family != 'unknown':
                instruments_set.add(family)
        
        # Сортируем для consistency
        instruments_list = sorted(list(instruments_set))
        
        # Создаем словарь с номерами
        instruments_dict = {}
        for idx, instrument in enumerate(instruments_list, 1):
            instruments_dict[idx] = {
                'id': idx,
                'name': instrument,
                'display_name': self.get_display_name(instrument)
            }
        
        return instruments_dict
    
    def get_display_name(self, instrument_name):
        """Получение читаемого имени инструмента"""
        display_names = {
            'bass': 'Бас (электронный/акустический)',
            'brass': 'Медные духовые',
            'flute': 'Флейта',
            'guitar': 'Гитара',
            'keyboard': 'Клавишные',
            'mallet': 'Ударные (маримба, ксилофон)',
            'organ': 'Орган',
            'reed': 'Деревянные духовые (кларнет, саксофон)',
            'string': 'Струнные (скрипка, виолончель)',
            'synth_lead': 'Синтезатор (лид)',
            'vocal': 'Вокал'
        }
        
        return display_names.get(instrument_name, instrument_name)
    
    def prepare_latent_vectors(self):
        """Подготовка средних латентных векторов для каждого инструмента"""
        from src.data.dataset import AudioDataLoader
        
        spectrograms_path = self.data_dir / "melspectrograms.npy"
        metadata_path = self.data_dir / "processed_metadata.json"
        
        # Создаем DataLoader для всех данных
        train_loader, _, _ = AudioDataLoader.create_dataloaders(
            spectrograms_path=str(spectrograms_path),
            metadata_path=str(metadata_path),
            batch_size=32,
            train_ratio=1.0,  # Используем все данные
            shuffle=False,
            num_workers=0
        )
        
        # Загружаем метаданные для получения меток
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Группируем индексы по инструментам
        instrument_indices = {}
        for idx, item in enumerate(metadata):
            family = item.get('instrument_family', 'unknown')
            if family in [inst['name'] for inst in self.instruments.values()]:
                if family not in instrument_indices:
                    instrument_indices[family] = []
                instrument_indices[family].append(idx)
        
        # Вычисляем средние латентные векторы для каждого инструмента
        latent_vectors = {}
        
        with torch.no_grad():
            for instrument_name, indices in instrument_indices.items():
                if not indices:
                    continue
                
                # Берем до 20 семплов для репрезентативности
                sample_indices = indices[:min(20, len(indices))]
                vectors = []
                
                for idx in tqdm(sample_indices, desc=f"  {instrument_name}", leave=False):
                    x, _ = train_loader.dataset[idx]
                    x = x.unsqueeze(0).to(self.device)
                    
                    mu, _ = self.model.encode(x)
                    vectors.append(mu.cpu().numpy())
                
                if vectors:
                    latent_vectors[instrument_name] = np.mean(vectors, axis=0)
        
        return latent_vectors
    
    def display_instruments_menu(self):
        """Отображение меню выбора инструментов"""
        print("\n" + "=" * 60)
        print("ДОСТУПНЫЕ ИНСТРУМЕНТЫ:")
        print("=" * 60)
        
        for idx, info in self.instruments.items():
            print(f"{idx:2d}. {info['display_name']}")
        
        print("-" * 60)
        print("Команды:")
        print("  select <номер1> <номер2> - выбрать инструменты")
        print("  list                    - показать список инструментов")
        print("  params                  - показать текущий выбор")
        print("  generate                - создать гибрид")
        print("  exit                    - выход")
        print("=" * 60)
    
    def select_instruments(self, instrument1_num, instrument2_num):
        """Выбор двух инструментов для гибридизации"""
        try:
            inst1_num = int(instrument1_num)
            inst2_num = int(instrument2_num)
            
            if inst1_num not in self.instruments or inst2_num not in self.instruments:
                print("Ошибка: номера инструментов должны быть в диапазоне 1-11")
                return False
            
            if inst1_num == inst2_num:
                print("Предупреждение: вы выбрали один и тот же инструмент дважды")
            
            inst1 = self.instruments[inst1_num]
            inst2 = self.instruments[inst2_num]
            
            self.current_selection = {
                'instrument1': inst1,
                'instrument2': inst2,
                'alpha': 0.5,  # Значение по умолчанию
                'interpolation': 'linear'
            }
            
            print(f"\n✓ Выбраны инструменты:")
            print(f"  1. {inst1['display_name']}")
            print(f"  2. {inst2['display_name']}")
            print(f"  Коэффициент смешивания (alpha): 0.5 (по умолчанию)")
            
            return True
            
        except ValueError:
            print("Ошибка: номера инструментов должны быть числами")
            return False
    
    def set_parameters(self):
        """Установка параметров гибридизации"""
        if not self.current_selection:
            print("Сначала выберите инструменты командой 'select'")
            return
        
        print("\nНастройка параметров гибридизации:")
        print(f"Текущий выбор: {self.current_selection['instrument1']['display_name']} + "
              f"{self.current_selection['instrument2']['display_name']}")
        
        # Запрос коэффициента смешивания
        while True:
            try:
                alpha_input = input("Коэффициент смешивания (0.0-1.0, или несколько через запятую): ").strip()
                
                if ',' in alpha_input:
                    # Несколько значений
                    alphas = [float(x.strip()) for x in alpha_input.split(',')]
                    if all(0 <= a <= 1 for a in alphas):
                        self.current_selection['alpha'] = alphas
                        print(f"Установлено {len(alphas)} значений alpha: {alphas}")
                        break
                    else:
                        print("Все значения должны быть в диапазоне 0.0-1.0")
                else:
                    # Одно значение
                    alpha = float(alpha_input)
                    if 0 <= alpha <= 1:
                        self.current_selection['alpha'] = alpha
                        print(f"Установлен alpha: {alpha}")
                        break
                    else:
                        print("Alpha должен быть в диапазоне 0.0-1.0")
                        
            except ValueError:
                print("Ошибка: введите число или числа через запятую")
        
        # Запрос метода интерполяции
        print("\nМетод интерполяции:")
        print("  1. Линейная (linear) - равномерное смешивание")
        print("  2. Сферическая (spherical) - учитывает геометрию пространства")
        
        while True:
            choice = input("Выберите метод (1 или 2): ").strip()
            if choice == '1':
                self.current_selection['interpolation'] = 'linear'
                print("Выбрана линейная интерполяция")
                break
            elif choice == '2':
                self.current_selection['interpolation'] = 'spherical'
                print("Выбрана сферическая интерполяция")
                break
            else:
                print("Пожалуйста, выберите 1 или 2")
    
    def generate_hybrid(self):
        """Генерация гибрида на основе текущего выбора"""
        if not self.current_selection:
            print("Сначала выберите инструменты командой 'select'")
            return
        
        inst1 = self.current_selection['instrument1']['name']
        inst2 = self.current_selection['instrument2']['name']
        alpha = self.current_selection['alpha']
        interpolation = self.current_selection['interpolation']
        
        # Получаем латентные векторы
        if inst1 not in self.latent_vectors or inst2 not in self.latent_vectors:
            print(f"Ошибка: латентные векторы не найдены для одного из инструментов")
            return
        
        z1 = self.latent_vectors[inst1]
        z2 = self.latent_vectors[inst2]
        
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / f"{inst1}_{inst2}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nСоздание гибрида...")
        print(f"  Инструмент 1: {self.current_selection['instrument1']['display_name']}")
        print(f"  Инструмент 2: {self.current_selection['instrument2']['display_name']}")
        print(f"  Метод интерполяции: {interpolation}")
        
        # Сохраняем информацию о выборе
        selection_info = {
            'timestamp': timestamp,
            'instruments': {
                'instrument1': self.current_selection['instrument1'],
                'instrument2': self.current_selection['instrument2']
            },
            'parameters': {
                'alpha': alpha if isinstance(alpha, list) else [alpha],
                'interpolation': interpolation
            },
            'model_info': {
                'latent_dim': self.checkpoint['config']['latent_dim'],
                'beta': self.checkpoint['config']['beta']
            }
        }
        
        with open(output_dir / "selection_info.json", 'w', encoding='utf-8') as f:
            json.dump(selection_info, f, indent=2, ensure_ascii=False)
        
        # Обрабатываем одно или несколько значений alpha
        if isinstance(alpha, list):
            alphas = alpha
        else:
            alphas = [alpha]
        
        # Генерируем гибриды для каждого alpha
        generated_specs = []
        
        for a in alphas:
            print(f"  Генерация гибрида с alpha={a:.3f}...")
            
            # Интерполяция в латентном пространстве
            if interpolation == 'linear':
                z_hybrid = self.linear_interpolation(z1, z2, a)
            else:  # spherical
                z_hybrid = self.spherical_interpolation(z1, z2, a)
            
            # Декодирование
            with torch.no_grad():
                z_tensor = torch.FloatTensor(z_hybrid).to(self.device)
                spectrogram = self.model.decode(z_tensor)
                spec_np = spectrogram.cpu().numpy()
                generated_specs.append((a, spec_np))
        
        # Визуализация и сохранение результатов
        self.save_results(generated_specs, output_dir, inst1, inst2)
        
        print(f"\n✓ Гибрид(ы) созданы и сохранены в: {output_dir}")
        print(f"  Файлы:")
        for file in output_dir.glob("*"):
            if file.is_file():
                print(f"    - {file.name}")
        
        # Предлагаем создать аудио
        self.offer_audio_generation(output_dir, generated_specs)
    
    def linear_interpolation(self, z1, z2, alpha):
        """Линейная интерполяция"""
        return (1 - alpha) * z1 + alpha * z2
    
    def spherical_interpolation(self, z1, z2, alpha):
        """Сферическая интерполяция (slerp)"""
        # Нормализуем векторы
        z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
        z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)
        
        # Вычисляем угол между векторами
        dot = np.clip(np.dot(z1_norm.flatten(), z2_norm.flatten()), -1, 1)
        omega = np.arccos(dot)
        
        if omega < 1e-10:
            return z1
        
        # Интерполяция
        interp = (np.sin((1 - alpha) * omega) * z1 + 
                  np.sin(alpha * omega) * z2) / np.sin(omega)
        
        return interp
    
    def save_results(self, generated_specs, output_dir, inst1, inst2):
        """Сохранение результатов гибридизации"""
        # Создаем визуализации
        n_specs = len(generated_specs)
        
        if n_specs == 1:
            # Один гибрид - подробная визуализация
            alpha, spec = generated_specs[0]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Спектрограмма гибрида
            axes[0].imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title(f"Гибрид: {inst1} + {inst2}\nalpha={alpha:.3f}")
            axes[0].set_xlabel("Время (кадры)")
            axes[0].set_ylabel("Мел-банки")
            
            # Информационная панель
            axes[1].axis('off')
            info_text = (
                f"ИНФОРМАЦИЯ О ГИБРИДЕ\n\n"
                f"Инструмент 1: {inst1}\n"
                f"  Доля: {(1-alpha):.1%}\n\n"
                f"Инструмент 2: {inst2}\n"
                f"  Доля: {alpha:.1%}\n\n"
                f"Параметры модели:\n"
                f"  Латентность: {self.checkpoint['config']['latent_dim']}\n"
                f"  Beta: {self.checkpoint['config']['beta']}\n\n"
                f"Размер спектрограммы: {spec.shape}"
            )
            axes[1].text(0.1, 0.5, info_text, fontsize=11, 
                        verticalalignment='center', linespacing=1.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / "hybrid_spectrogram.png", dpi=150)
            plt.close()
            
            # Сохраняем спектрограмму как numpy файл
            np.save(output_dir / "hybrid_spectrogram.npy", spec)
            
        else:
            # Несколько гибридов - сравнение
            fig, axes = plt.subplots(n_specs, 1, figsize=(10, 3 * n_specs))
            
            if n_specs == 1:
                axes = [axes]
            
            for idx, (alpha, spec) in enumerate(generated_specs):
                ax = axes[idx]
                ax.imshow(spec[0, 0], aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f"Гибрид {idx+1}: alpha={alpha:.3f} "
                           f"({(1-alpha):.1%} {inst1} + {alpha:.1%} {inst2})")
                ax.set_xlabel("Время (кадры)")
                ax.set_ylabel("Мел-банки")
            
            plt.suptitle(f"Сравнение гибридов: {inst1} ↔ {inst2}", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "hybrids_comparison.png", dpi=150)
            plt.close()
            
            # Сохраняем все спектрограммы
            for idx, (alpha, spec) in enumerate(generated_specs):
                np.save(output_dir / f"hybrid_alpha_{alpha:.3f}.npy", spec)
    
    def offer_audio_generation(self, output_dir, generated_specs):
        """Предложение сгенерировать аудио из спектрограмм"""
        print("\n" + "-" * 60)
        print("Хотите сгенерировать аудиофайлы из спектрограмм?")
        print("Это займет некоторое время и качество будет приблизительным.")
        
        while True:
            choice = input("Сгенерировать аудио? (y/n): ").lower().strip()
            
            if choice in ['y', 'yes', 'д', 'да']:
                self.generate_audio_files(output_dir, generated_specs)
                break
            elif choice in ['n', 'no', 'н', 'нет']:
                print("Аудио не будет сгенерировано.")
                break
            else:
                print("Пожалуйста, введите 'y' или 'n'")
    
    def generate_audio_files(self, output_dir, generated_specs):
        """Генерация аудиофайлов из спектрограмм"""
        print("\nГенерация аудиофайлов...")
        
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for alpha, spec in generated_specs:
            audio_path = audio_dir / f"hybrid_alpha_{alpha:.3f}.wav"
            
            try:
                # Преобразование спектрограммы в аудио
                success = self.spectrogram_to_audio(spec[0, 0], audio_path)
                
                if success:
                    print(f"  ✓ Аудио сохранено: {audio_path.name}")
                else:
                    print(f"  ✗ Не удалось сгенерировать аудио для alpha={alpha:.3f}")
                    
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
        
        print(f"\nАудиофайлы сохранены в: {audio_dir}")
    
    def spectrogram_to_audio(self, spectrogram, output_path, sr=16000):
        """Преобразование спектрограммы в аудио (упрощенное)"""
        try:
            # Обратное преобразование мел-спектрограммы
            n_fft = 2048
            hop_length = 512
            
            # Преобразуем из нормализованного вида
            # Предполагаем, что спектрограмма нормализована к [0, 1]
            spec_db = spectrogram * 80 - 40  # Примерное преобразование к dB
            
            # Преобразуем из dB к мощности
            spec_power = librosa.db_to_power(spec_db)
            
            # Обратное мел-преобразование (приблизительное)
            mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
            
            # Псевдообратная матрица
            inv_mel_basis = np.linalg.pinv(mel_basis)
            
            # Восстанавливаем спектрограмму
            spec_mag = np.dot(inv_mel_basis, spec_power)
            
            # Добавляем случайную фазу (упрощенный Griffin-Lim)
            phase = np.exp(1j * np.random.uniform(0, 2*np.pi, size=spec_mag.shape))
            stft_matrix = spec_mag * phase
            
            # Обратное STFT
            y = librosa.istft(stft_matrix, hop_length=hop_length, win_length=n_fft)
            
            # Нормализация
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y)) * 0.8
            
            # Сохраняем аудио
            sf.write(output_path, y, sr)
            return True
            
        except Exception as e:
            print(f"    Ошибка преобразования: {e}")
            return False
    
    def show_current_selection(self):
        """Показать текущий выбор инструментов и параметров"""
        if not self.current_selection:
            print("Инструменты еще не выбраны. Используйте команду 'select'.")
            return
        
        inst1 = self.current_selection['instrument1']
        inst2 = self.current_selection['instrument2']
        
        print("\nТЕКУЩИЙ ВЫБОР:")
        print(f"  Инструмент 1: {inst1['display_name']} (ID: {inst1['id']})")
        print(f"  Инструмент 2: {inst2['display_name']} (ID: {inst2['id']})")
        
        alpha = self.current_selection['alpha']
        if isinstance(alpha, list):
            print(f"  Коэффициенты смешивания: {', '.join([f'{a:.3f}' for a in alpha])}")
        else:
            print(f"  Коэффициент смешивания: {alpha}")
        
        print(f"  Метод интерполяции: {self.current_selection['interpolation']}")
    
    def run_interactive_loop(self):
        """Запуск интерактивного цикла"""
        print("\nДобро пожаловать в интерактивный гибридизатор тембров!")
        print("Выберите два инструмента по номерам и создайте уникальный гибрид.")
        
        self.display_instruments_menu()
        
        while True:
            try:
                command = input("\n>>> ").strip().lower()
                
                if command == 'exit' or command == 'quit' or command == 'выход':
                    print("\nВыход из программы. До свидания!")
                    break
                
                elif command == 'list':
                    self.display_instruments_menu()
                
                elif command.startswith('select '):
                    parts = command.split()
                    if len(parts) == 3:
                        if self.select_instruments(parts[1], parts[2]):
                            print("\nТеперь настройте параметры командой 'params' "
                                  "или сразу создайте гибрид командой 'generate'")
                    else:
                        print("Использование: select <номер1> <номер2>")
                        print("Пример: select 3 7")
                
                elif command == 'params':
                    self.set_parameters()
                
                elif command == 'generate':
                    if self.current_selection:
                        self.generate_hybrid()
                    else:
                        print("Сначала выберите инструменты командой 'select'")
                
                elif command == 'current' or command == 'status':
                    self.show_current_selection()
                
                elif command == 'help' or command == 'помощь':
                    print("\nДоступные команды:")
                    print("  select <n1> <n2> - выбрать два инструмента по номерам")
                    print("  params           - настроить параметры смешивания")
                    print("  generate         - создать гибрид")
                    print("  list             - показать список инструментов")
                    print("  current          - показать текущий выбор")
                    print("  exit             - выход из программы")
                
                else:
                    print(f"Неизвестная команда: {command}")
                    print("Введите 'help' для списка команд")
                    
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем. Выход...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")

def main():
    parser = argparse.ArgumentParser(description='Интерактивный гибридизатор тембров')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/final_vae_150epochs/best_model.pth',
                       help='Путь к чекпоинту модели')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для вычислений (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Проверяем наличие файлов
    if not Path(args.checkpoint).exists():
        print(f"Ошибка: чекпоинт не найден: {args.checkpoint}")
        print("Сначала обучите модель или укажите правильный путь.")
        return
    
    if not Path(args.data_dir).exists():
        print(f"Ошибка: данные не найдены: {args.data_dir}")
        return
    
    # Запускаем интерактивный гибридизатор
    hybridizer = InteractiveHybridizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )
    
    hybridizer.run_interactive_loop()

if __name__ == "__main__":
    main()