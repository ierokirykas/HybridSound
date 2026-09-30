# scripts/hyperparameter_search.py
#!/usr/bin/env python3
"""Гиперпараметрический поиск для VAE модели"""

import sys
from pathlib import Path
import argparse
import json
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import subprocess
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class HyperparameterSearch:
    def __init__(self, config_path="config/hyperparam_config.yaml"):
        """Инициализация поиска гиперпараметров"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['hyperparam_search']
        
        # Директория для результатов
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем таблицу результатов
        self.results_file = self.results_dir / "search_results.csv"
        self.summary_file = self.results_dir / "search_summary.json"
        
        # Параметры для поиска
        self.param_grid = self.config['param_grid']
        
        print("=" * 70)
        print("ГИПЕРПАРАМЕТРИЧЕСКИЙ ПОИСК ДЛЯ VAE")
        print("=" * 70)
        
    def generate_param_combinations(self):
        """Генерация всех комбинаций параметров"""
        param_names = []
        param_values = []
        
        for param, values in self.param_grid.items():
            param_names.append(param)
            if isinstance(values, list):
                param_values.append(values)
            else:
                # Если указан диапазон [min, max, step]
                if len(values) == 3:
                    start, stop, step = values
                    param_values.append(list(np.arange(start, stop + step, step)))
                else:
                    param_values.append(values)
        
        # Генерируем все комбинации
        all_combinations = list(itertools.product(*param_values))
        
        # Создаем список словарей
        param_list = []
        for combo in all_combinations:
            param_dict = dict(zip(param_names, combo))
            param_list.append(param_dict)
        
        print(f"Сгенерировано {len(param_list)} комбинаций параметров")
        return param_list
    
    def create_experiment_config(self, params, exp_id):
        """Создание конфигурационного файла для эксперимента"""
        # Базовый конфиг модели
        model_config = {
            'latent_dim': int(params['latent_dim']),
            'beta': float(params['beta']),
            'learning_rate': float(params['learning_rate']),
            'batch_size': int(params['batch_size']),
            'attr_weights': {
                'instrument_family': float(params.get('weight_family', 0.3)),
                'instrument_source': 0.1,
                'brightness': 0.05,
                'attack_type': 0.05
            }
        }
        
        # Создаем директорию для эксперимента
        exp_dir = self.results_dir / f"exp_{exp_id:03d}"
        exp_dir.mkdir(exist_ok=True)
        
        # Сохраняем конфиг
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump({'model': model_config}, f)
        
        return exp_dir, config_path
    
    def run_experiment(self, params, exp_id, data_dir):
        """Запуск одного эксперимента"""
        print(f"\nЭксперимент {exp_id}:")
        print(f"  latent_dim: {params['latent_dim']}")
        print(f"  beta: {params['beta']}")
        print(f"  lr: {params['learning_rate']}")
        print(f"  batch_size: {params['batch_size']}")
        
        # Создаем конфиг для эксперимента
        exp_dir, config_path = self.create_experiment_config(params, exp_id)
        
        # Команда для запуска обучения
        cmd = [
            'python', 'scripts/train_vae.py',
            '--data_dir', str(data_dir),
            '--batch_size', str(int(params['batch_size'])),
            '--latent_dim', str(int(params['latent_dim'])),
            '--epochs', '20',
            '--lr', str(float(params['learning_rate'])),
            '--beta', str(float(params['beta'])),
            '--checkpoint_dir', str(exp_dir),
            '--device', 'cuda',
            '--grad_clip', '1.0'
        ]
        
        # Запускаем обучение
        print(f"  Запуск обучения...")
        start_time = time.time()
        # print(cmd)
        try:
            # Запускаем процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # Ждем завершения
            stdout, stderr = process.communicate()
            
            # Сохраняем лог
            log_file = exp_dir / "training_log.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n\nSTDERR:\n")
                f.write(stderr)
            
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                print(f"  ✓ Обучение завершено за {elapsed_time:.1f} сек")
                
                # Пытаемся извлечь результаты из лога
                results = self.extract_results_from_log(stdout, exp_dir)
                results['exp_id'] = exp_id
                results['params'] = params
                results['elapsed_time'] = elapsed_time
                results['status'] = 'success'
                
                return results
            else:
                print(f"  ✗ Ошибка при обучении (код {process.returncode})")
                if stderr:
                    print(f"    Ошибка: {stderr[:200]}...")
                
                return {
                    'exp_id': exp_id,
                    'params': params,
                    'elapsed_time': elapsed_time,
                    'status': 'failed',
                    'error': stderr[:500]
                }
                
        except Exception as e:
            print(f"  ✗ Исключение: {e}")
            return {
                'exp_id': exp_id,
                'params': params,
                'elapsed_time': time.time() - start_time,
                'status': 'exception',
                'error': str(e)
            }
    
    def extract_results_from_log(self, log_text, exp_dir):
        """Извлечение результатов из лога обучения"""
        results = {}
        
        try:
            # Ищем лучшую val loss
            import re
            
            # Паттерны для поиска
            patterns = {
                'best_val_loss': r'Лучшая val loss: (\d+\.\d+)',
                'final_train_loss': r'Train Loss: (\d+\.\d+)',
                'final_val_loss': r'Val Loss: (\d+\.\d+)',
                'final_recon': r'Recon: (\d+\.\d+)',
                'final_kl': r'KL: (\d+\.\d+)',
                'final_attr': r'Attr: (\d+\.\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, log_text)
                if match:
                    results[key] = float(match.group(1))
            
            # Пытаемся загрузить историю обучения
            history_file = exp_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if 'train_loss' in history and len(history['train_loss']) > 0:
                    results['final_epoch'] = len(history['train_loss'])
                    results['min_train_loss'] = min(history['train_loss'])
                    results['min_val_loss'] = min(history['val_loss'])
            
            # Пытаемся проанализировать модель
            checkpoint_file = exp_dir / "best_model.pth"
            if checkpoint_file.exists():
                import torch
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                results['best_epoch'] = checkpoint.get('epoch', 'unknown')
                
                # Запускаем быстрый анализ модели
                analysis_results = self.quick_analysis(exp_dir)
                if analysis_results:
                    results.update(analysis_results)
            
        except Exception as e:
            print(f"    Предупреждение: не удалось извлечь все результаты: {e}")
        
        return results
    
    def quick_analysis(self, exp_dir):
        """Быстрый анализ обученной модели"""
        try:
            # Импортируем здесь, чтобы избежать циклических импортов
            sys.path.append(str(project_root))
            
            from scripts.analyze_vae import load_model, load_data
            
            # Загружаем модель
            checkpoint_path = exp_dir / "best_model.pth"
            data_dir = Path("data/processed/nsynth_train_processed")
            
            if not checkpoint_path.exists() or not data_dir.exists():
                return None
            
            # Загружаем модель (упрощенная версия)
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get('config', {})
            
            from src.models.fixed_vae import FixedSizeVAE
            model = FixedSizeVAE(
                latent_dim=config.get('latent_dim', 16),
                attribute_dims=config.get('attribute_dims', {}),
                beta=config.get('beta', 1.0)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Быстрая оценка на небольшом подмножестве
            from src.data.dataset import AudioDataLoader
            
            spectrograms_path = data_dir / "melspectrograms.npy"
            metadata_path = data_dir / "processed_metadata.json"
            
            # Создаем небольшой dataloader
            _, val_loader, _ = AudioDataLoader.create_dataloaders(
                spectrograms_path=str(spectrograms_path),
                metadata_path=str(metadata_path),
                batch_size=32,
                train_ratio=0.8,
                shuffle=False,
                num_workers=0
            )
            
            # Вычисляем accuracy для instrument_family
            import torch.nn.functional as F
            
            correct = 0
            total = 0
            latent_vectors = []
            labels = []
            
            with torch.no_grad():
                for x, attributes in val_loader:
                    x = x.to(device)
                    
                    # Forward pass
                    _, mu, _, attr_preds = model(x)
                    
                    # Accuracy для instrument_family
                    if 'instrument_family' in attr_preds and 'instrument_family' in attributes:
                        preds = torch.argmax(attr_preds['instrument_family'], dim=1)
                        targets = attributes['instrument_family'].to(device)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)
                    
                    # Сохраняем латентные векторы и метки
                    latent_vectors.append(mu.cpu().numpy())
                    if 'instrument_family' in attributes:
                        labels.extend(attributes['instrument_family'].cpu().numpy())
                    
                    # Ограничиваем количество батчей для скорости
                    if len(latent_vectors) >= 5:
                        break
            
            accuracy = correct / total if total > 0 else 0
            
            # Вычисляем Silhouette Score (на подмножестве)
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            if len(latent_vectors) > 0 and len(labels) > 1:
                latent_vectors_np = np.concatenate(latent_vectors, axis=0)
                labels_np = np.array(labels)
                
                # Берем только часть данных для скорости
                n_samples = min(500, len(labels_np))
                indices = np.random.choice(len(labels_np), n_samples, replace=False)
                
                try:
                    sil_score = silhouette_score(
                        latent_vectors_np[indices],
                        labels_np[indices]
                    )
                except:
                    sil_score = -1.0
            else:
                sil_score = -1.0
            
            # Вычисляем PSNR на нескольких примерах
            mse_values = []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x[:2].to(device)
                    x_recon, _, _, _ = model(x)
                    mse = F.mse_loss(x_recon, x, reduction='mean').item()
                    mse_values.append(mse)
                    break
            
            avg_mse = np.mean(mse_values) if mse_values else 0
            psnr = 20 * np.log10(1.0) - 10 * np.log10(avg_mse) if avg_mse > 0 else 0
            
            return {
                'instrument_family_accuracy': accuracy,
                'silhouette_score': sil_score,
                'mse': avg_mse,
                'psnr': psnr
            }
            
        except Exception as e:
            print(f"    Предупреждение: быстрый анализ не удался: {e}")
            return None
    
    def run_search(self, data_dir, max_experiments=None):
        """Запуск поиска гиперпараметров"""
        print(f"\nЗапуск гиперпараметрического поиска")
        print(f"Данные: {data_dir}")
        
        # Генерируем комбинации параметров
        param_combinations = self.generate_param_combinations()
        
        if max_experiments:
            param_combinations = param_combinations[:max_experiments]
        
        print(f"Будет выполнено {len(param_combinations)} экспериментов")
        
        # Запускаем эксперименты
        all_results = []
        failed_experiments = []
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}]", end="")
            
            results = self.run_experiment(params, i, data_dir)
            all_results.append(results)
            
            # Сохраняем промежуточные результаты
            self.save_intermediate_results(all_results, failed_experiments)
            
            # Небольшая пауза между экспериментами
            if i < len(param_combinations):
                time.sleep(2)
        
        # Сохраняем финальные результаты
        self.save_final_results(all_results, failed_experiments)
        
        return all_results
    
    def save_intermediate_results(self, results, failed):
        """Сохранение промежуточных результатов"""
        # Фильтруем успешные эксперименты
        successful = [r for r in results if r.get('status') == 'success']
        
        if successful:
            # Создаем DataFrame
            df_data = []
            for res in successful:
                row = res['params'].copy()
                row.update({k: v for k, v in res.items() 
                          if k not in ['params', 'status', 'error']})
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Сохраняем
            temp_file = self.results_dir / "temp_results.csv"
            df.to_csv(temp_file, index=False)
    
    def save_final_results(self, all_results, failed_experiments):
        """Сохранение финальных результатов"""
        print(f"\n\nСохранение результатов...")
        
        # Разделяем успешные и неуспешные эксперименты
        successful = [r for r in all_results if r.get('status') == 'success']
        failed = [r for r in all_results if r.get('status') != 'success']
        
        print(f"Успешных: {len(successful)}")
        print(f"Неуспешных: {len(failed)}")
        
        if successful:
            # Создаем DataFrame для успешных
            df_data = []
            for res in successful:
                row = res['params'].copy()
                row.update({k: v for k, v in res.items() 
                          if k not in ['params', 'status', 'error']})
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Сохраняем
            df.to_csv(self.results_file, index=False)
            print(f"✓ Результаты сохранены в {self.results_file}")
            
            # Анализ лучших моделей
            self.analyze_best_models(df)
        
        # Сохраняем сводку
        summary = {
            'total_experiments': len(all_results),
            'successful': len(successful),
            'failed': len(failed),
            'search_completed': datetime.now().isoformat(),
            'best_models': self.get_best_models_summary(successful)
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Сводка сохранена в {self.summary_file}")
    
    def analyze_best_models(self, df):
        """Анализ лучших моделей по разным метрикам"""
        print(f"\nАнализ лучших моделей:")
        
        if 'instrument_family_accuracy' in df.columns:
            best_acc = df.loc[df['instrument_family_accuracy'].idxmax()]
            print(f"  Лучшая accuracy: {best_acc['instrument_family_accuracy']:.3f}")
            print(f"    Параметры: latent_dim={best_acc['latent_dim']}, "
                  f"beta={best_acc['beta']}, lr={best_acc['learning_rate']}")
        
        if 'silhouette_score' in df.columns:
            best_sil = df.loc[df['silhouette_score'].idxmax()]
            print(f"  Лучший Silhouette Score: {best_sil['silhouette_score']:.3f}")
            print(f"    Параметры: latent_dim={best_sil['latent_dim']}, "
                  f"beta={best_sil['beta']}, lr={best_sil['learning_rate']}")
        
        if 'psnr' in df.columns:
            best_psnr = df.loc[df['psnr'].idxmax()]
            print(f"  Лучший PSNR: {best_psnr['psnr']:.2f} dB")
            print(f"    Параметры: latent_dim={best_psnr['latent_dim']}, "
                  f"beta={best_psnr['beta']}, lr={best_psnr['learning_rate']}")
        
        # Композитная метрика
        if all(k in df.columns for k in ['psnr', 'instrument_family_accuracy']):
            # Нормализуем метрики
            df_norm = df.copy()
            df_norm['psnr_norm'] = (df['psnr'] - df['psnr'].min()) / (df['psnr'].max() - df['psnr'].min())
            df_norm['acc_norm'] = (df['instrument_family_accuracy'] - df['instrument_family_accuracy'].min()) / \
                                 (df['instrument_family_accuracy'].max() - df['instrument_family_accuracy'].min())
            
            # Композитная метрика
            df_norm['composite_score'] = df_norm['psnr_norm'] + df_norm['acc_norm']
            
            best_composite = df_norm.loc[df_norm['composite_score'].idxmax()]
            print(f"\n  Лучшая композитная метрика: {best_composite['composite_score']:.3f}")
            print(f"    PSNR: {best_composite['psnr']:.2f} dB")
            print(f"    Accuracy: {best_composite['instrument_family_accuracy']:.3f}")
            print(f"    Параметры: latent_dim={best_composite['latent_dim']}, "
                  f"beta={best_composite['beta']}, lr={best_composite['learning_rate']}")
    
    def get_best_models_summary(self, successful_results):
        """Получение сводки по лучшим моделям"""
        if not successful_results:
            return {}
        
        # Создаем DataFrame из результатов
        df_data = []
        for res in successful_results:
            if 'params' in res:
                row = res['params'].copy()
                row.update({k: v for k, v in res.items() 
                          if k not in ['params', 'status', 'error']})
                df_data.append(row)
        
        if not df_data:
            return {}
        
        df = pd.DataFrame(df_data)
        
        summary = {}
        
        # Лучшие по каждой метрике
        metrics = ['instrument_family_accuracy', 'silhouette_score', 'psnr', 'best_val_loss']
        for metric in metrics:
            if metric in df.columns and not df[metric].isnull().all():
                try:
                    if metric == 'best_val_loss':
                        # Для loss меньше - лучше
                        best_idx = df[metric].idxmin()
                    else:
                        # Для других метрик больше - лучше
                        best_idx = df[metric].idxmax()
                    
                    best_row = df.loc[best_idx]
                    summary[f'best_{metric}'] = {
                        'value': float(best_row[metric]),
                        'latent_dim': int(best_row['latent_dim']),
                        'beta': float(best_row['beta']),
                        'learning_rate': float(best_row['learning_rate']),
                        'batch_size': int(best_row['batch_size'])
                    }
                except:
                    pass
        
        return summary
    
    def create_visualizations(self):
        """Создание визуализаций результатов поиска"""
        if not self.results_file.exists():
            print("Файл результатов не найден")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.read_csv(self.results_file)
            
            if len(df) < 2:
                print("Недостаточно данных для визуализации")
                return
            
            # Настройки стиля
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # 1. Heatmap: beta vs latent_dim
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if all(col in df.columns for col in ['beta', 'latent_dim', 'instrument_family_accuracy']):
                # Pivot table для heatmap
                pivot_table = df.pivot_table(
                    values='instrument_family_accuracy',
                    index='latent_dim',
                    columns='beta',
                    aggfunc='mean'
                )
                
                # Heatmap
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                           ax=axes[0, 0], cbar_kws={'label': 'Accuracy'})
                axes[0, 0].set_title('Accuracy по instrument_family\n(beta vs latent_dim)')
                axes[0, 0].set_xlabel('Beta')
                axes[0, 0].set_ylabel('Latent Dimension')
            
            # 2. Scatter: PSNR vs Accuracy
            if all(col in df.columns for col in ['psnr', 'instrument_family_accuracy']):
                scatter = axes[0, 1].scatter(
                    df['psnr'], 
                    df['instrument_family_accuracy'],
                    c=df['beta'],
                    s=df['latent_dim']*5,
                    alpha=0.6,
                    cmap='viridis'
                )
                
                axes[0, 1].set_xlabel('PSNR (dB)')
                axes[0, 1].set_ylabel('Instrument Family Accuracy')
                axes[0, 1].set_title('Trade-off: Качество vs Распознавание')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Цветовая шкала для beta
                plt.colorbar(scatter, ax=axes[0, 1], label='Beta')
                
                # Легенда для размера точек (latent_dim)
                handles = []
                unique_dims = sorted(df['latent_dim'].unique())
                for dim in unique_dims:
                    handles.append(plt.scatter([], [], s=dim*5, alpha=0.6, 
                                              color='gray', label=f'latent={dim}'))
                axes[0, 1].legend(handles=handles, title='Latent Dim', 
                                 loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # 3. Влияние beta на разные метрики
            if 'beta' in df.columns:
                metrics_to_plot = ['instrument_family_accuracy', 'silhouette_score', 'psnr']
                
                for i, metric in enumerate(metrics_to_plot):
                    if metric in df.columns:
                        ax = axes[1, i]
                        
                        # Group by beta
                        grouped = df.groupby('beta')[metric].agg(['mean', 'std']).reset_index()
                        
                        ax.errorbar(
                            grouped['beta'],
                            grouped['mean'],
                            yerr=grouped['std'],
                            marker='o',
                            capsize=5,
                            linestyle='-'
                        )
                        
                        ax.set_xlabel('Beta')
                        ax.set_ylabel(metric.replace('_', ' ').title())
                        ax.set_title(f'Влияние beta на {metric}')
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Визуализации сохранены в {self.results_dir}")
            
        except Exception as e:
            print(f"✗ Ошибка при создании визуализаций: {e}")

def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(description='Гиперпараметрический поиск для VAE')
    parser.add_argument('--data_dir', type=str, 
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--config', type=str, 
                       default='config/hyperparam_config.yaml',
                       help='Конфигурационный файл поиска')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Максимальное количество экспериментов')
    parser.add_argument('--visualize', action='store_true',
                       help='Создать визуализации после поиска')
    
    args = parser.parse_args()
    
    # Проверяем существование конфигурационного файла
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Конфигурационный файл не найден: {config_path}")
        print("Создаю базовый конфиг...")
        
        # Создаем директорию для конфига
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Базовый конфиг
        base_config = """hyperparam_search:
  results_dir: "hyperparam_search_results"
  
  param_grid:
    latent_dim: [16, 32, 64]
    beta: [0.1, 0.5, 1.0, 2.0, 5.0]
    learning_rate: [0.001, 0.0005, 0.0001]
    batch_size: [32, 64]
    weight_family: [0.3, 0.5]
"""
        with open(config_path, 'w') as f:
            yaml.dump(yaml.safe_load(base_config), f)
        
        print(f"✓ Создан конфиг: {config_path}")
        print("При необходимости отредактируйте файл и запустите скрипт снова.")
        return
    
    # Проверяем существование данных
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Ошибка: директория с данными не найдена: {data_dir}")
        return
    
    # Запускаем поиск
    searcher = HyperparameterSearch(args.config)
    results = searcher.run_search(str(data_dir), args.max_experiments)
    
    # Создаем визуализации
    if args.visualize or True:  # Всегда создаем визуализации
        searcher.create_visualizations()
    
    print("\n" + "=" * 70)
    print("ГИПЕРПАРАМЕТРИЧЕСКИЙ ПОИСК ЗАВЕРШЕН!")
    print("=" * 70)
    print(f"Результаты сохранены в: {searcher.results_dir}")
    print(f"Для просмотра результатов выполните:")
    print(f"  python -c \"import pandas as pd; df = pd.read_csv('{searcher.results_file}'); print(df[['latent_dim', 'beta', 'instrument_family_accuracy', 'psnr']].head())\"")

if __name__ == "__main__":
    main()