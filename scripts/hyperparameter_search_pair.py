# scripts/hyperparameter_search_pair.py
#!/usr/bin/env python3
"""
Гиперпараметрический поиск для VAE модели, обученной на паре инструментов.
"""

import sys
from pathlib import Path
import argparse
import yaml
import itertools
import numpy as np
from tqdm import tqdm
import subprocess
import time
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class PairHyperparameterSearch:
    def __init__(self, config_path):
        """Инициализация поиска гиперпараметров для пары инструментов"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['hyperparam_pair_search']
        
        # Пара инструментов
        self.instrument1 = self.config['instrument_pair']['instrument1']
        self.instrument2 = self.config['instrument_pair']['instrument2']
        self.pair_name = self.config['instrument_pair']['name']
        
        # Директория для результатов
        self.results_dir = Path(self.config['results_dir']) / self.pair_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Сетка параметров
        self.param_grid = self.config['param_grid']
        
        # Создаем таблицу результатов
        self.results_file = self.results_dir / "search_results.csv"
        self.summary_file = self.results_dir / "search_summary.json"
        
        print("=" * 70)
        print(f"ГИПЕРПАРАМЕТРИЧЕСКИЙ ПОИСК ДЛЯ ПАРЫ: {self.instrument1} + {self.instrument2}")
        print("=" * 70)
    
    def generate_param_combinations(self):
        """Генерация всех комбинаций параметров"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
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
        # Базовый конфиг
        exp_config = {
            'hyperparam_pair_search': {
                'results_dir': str(self.results_dir),
                'instrument_pair': {
                    'name': self.pair_name,
                    'instrument1': self.instrument1,
                    'instrument2': self.instrument2
                },
                'param_grid': self.param_grid,
                'training': self.config['training'],
                'metrics': self.config['metrics']
            }
        }
        
        # Создаем директорию для эксперимента
        exp_dir = self.results_dir / f"exp_{exp_id:03d}"
        exp_dir.mkdir(exist_ok=True)
        
        # Сохраняем конфиг
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(exp_config, f)
        
        return exp_dir, config_path
    
    def run_experiment(self, params, exp_id, data_dir, device):
        """Запуск одного эксперимента"""
        print(f"\nЭксперимент {exp_id}:")
        print(f"  latent_dim: {params['latent_dim']}")
        print(f"  beta: {params['beta']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  batch_size: {params['batch_size']}")
        
        # Создаем конфиг для эксперимента
        exp_dir, config_path = self.create_experiment_config(params, exp_id)
        
        # Команда для запуска обучения
        cmd = [
            'python', 'scripts/train_pair_vae.py',
            '--config', str(config_path),
            '--data_dir', data_dir,
            '--device', device
        ]
        
        # Дополнительные параметры (передаем через аргументы или модифицируем конфиг)
        # В данном случае параметры уже в конфиге, но train_pair_vae.py ожидает отдельные аргументы
        # Поэтому создадим временный конфиг с конкретными параметрами
        
        # Создаем конкретный конфиг для этого эксперимента
        specific_config = {
            'data_dir': data_dir,
            'instrument1': self.instrument1,
            'instrument2': self.instrument2,
            'latent_dim': params['latent_dim'],
            'beta': params['beta'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'epochs': self.config['training']['epochs'],
            'device': device,
            'output_dir': str(exp_dir)
        }
        
        # Сохраняем конкретный конфиг
        specific_config_path = exp_dir / "train_config.yaml"
        with open(specific_config_path, 'w') as f:
            yaml.dump(specific_config, f)
        
        # Модифицируем команду
        cmd = [
            'python', '-c',
            f'''
import sys
sys.path.append("{project_root}")
from scripts.train_pair_vae import train_pair_vae
import yaml

with open("{specific_config_path}", "r") as f:
    config = yaml.safe_load(f)

train_pair_vae(config)
            '''
        ]
        
        # Запускаем обучение
        print(f"  Запуск обучения...")
        start_time = time.time()
        
        try:
            # Запускаем процесс
            process = subprocess.Popen(
                ' '.join(cmd),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # Ждем завершения с таймаутом
            timeout = 1800  # 30 минут на эксперимент
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                print(f"  ⚠️  Таймаут ({timeout} секунд)")
            
            elapsed_time = time.time() - start_time
            
            # Сохраняем лог
            log_file = exp_dir / "training_log.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n\nSTDERR:\n")
                f.write(stderr)
            
            if process.returncode == 0:
                print(f"  ✓ Обучение завершено за {elapsed_time:.1f} сек")
                
                # Извлекаем результаты из JSON файла
                results_file = exp_dir / "results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Добавляем информацию об эксперименте
                    results['exp_id'] = exp_id
                    results['params'] = params
                    results['elapsed_time'] = elapsed_time
                    results['status'] = 'success'
                    
                    return results
                else:
                    print(f"  ✗ Файл результатов не найден")
                    return {
                        'exp_id': exp_id,
                        'params': params,
                        'elapsed_time': elapsed_time,
                        'status': 'error',
                        'error': 'results file not found'
                    }
            else:
                print(f"  ✗ Ошибка при обучении (код {process.returncode})")
                return {
                    'exp_id': exp_id,
                    'params': params,
                    'elapsed_time': elapsed_time,
                    'status': 'error',
                    'error': stderr[:500] if stderr else 'unknown error'
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
    
    def calculate_composite_score(self, metrics, params):
        """Вычисление композитной оценки модели"""
        # Веса из конфига
        recon_weight = self.config['metrics']['reconstruction_weight']
        smoothness_weight = self.config['metrics']['smoothness_weight']
        
        # Базовые метрики
        psnr = metrics.get('psnr_mean', 0)
        smoothness = metrics.get('smoothness_score', 0)
        
        # Нормализуем метрики (предполагаем диапазоны)
        # PSNR: 0-40 dB, чем больше - лучше
        psnr_norm = min(psnr / 40, 1.0)
        
        # Плавность: 0-0.1, чем меньше - лучше (более плавно)
        # Преобразуем: smoothness_score = 0.1 - actual_smoothness
        if smoothness > 0:
            smoothness_norm = max(0, 1.0 - (smoothness * 10))  # Предполагаем smoothness < 0.1
        else:
            smoothness_norm = 0
        
        # Композитная оценка
        composite_score = (recon_weight * psnr_norm + 
                          smoothness_weight * smoothness_norm)
        
        return {
            'composite_score': composite_score,
            'psnr_norm': psnr_norm,
            'smoothness_norm': smoothness_norm,
            'psnr': psnr,
            'smoothness': smoothness
        }
    
    def run_search(self, data_dir, device='cuda', max_experiments=None):
        """Запуск поиска гиперпараметров"""
        print(f"\nЗапуск гиперпараметрического поиска")
        print(f"Данные: {data_dir}")
        print(f"Устройство: {device}")
        
        # Генерируем комбинации параметров
        param_combinations = self.generate_param_combinations()
        
        if max_experiments:
            param_combinations = param_combinations[:max_experiments]
        
        print(f"Будет выполнено {len(param_combinations)} экспериментов")
        
        # Запускаем эксперименты
        all_results = []
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}]", end="")
            
            results = self.run_experiment(params, i, data_dir, device)
            all_results.append(results)
            
            # Сохраняем промежуточные результаты
            self.save_intermediate_results(all_results)
            
            # Пауза между экспериментами
            if i < len(param_combinations):
                time.sleep(5)
        
        # Сохраняем финальные результаты
        self.save_final_results(all_results)
        
        # Анализируем результаты
        self.analyze_results(all_results)
        
        return all_results
    
    def save_intermediate_results(self, all_results):
        """Сохранение промежуточных результатов"""
        # Фильтруем успешные эксперименты
        successful = [r for r in all_results if r.get('status') == 'success']
        
        if successful:
            df_data = []
            for res in successful:
                if 'test_metrics' in res:
                    row = res['params'].copy()
                    row.update({
                        'exp_id': res['exp_id'],
                        'best_test_loss': res.get('best_test_loss', 0),
                        'psnr_mean': res['test_metrics'].get('psnr_mean', 0),
                        'smoothness_score': res['test_metrics'].get('smoothness_score', 0),
                        'elapsed_time': res.get('elapsed_time', 0)
                    })
                    
                    # Вычисляем композитную оценку
                    composite = self.calculate_composite_score(res['test_metrics'], res['params'])
                    row.update(composite)
                    
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                temp_file = self.results_dir / "temp_results.csv"
                df.to_csv(temp_file, index=False)
    
    def save_final_results(self, all_results):
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
                if 'test_metrics' in res:
                    row = res['params'].copy()
                    row.update({
                        'exp_id': res['exp_id'],
                        'best_test_loss': res.get('best_test_loss', 0),
                        'psnr_mean': res['test_metrics'].get('psnr_mean', 0),
                        'smoothness_score': res['test_metrics'].get('smoothness_score', 0),
                        'elapsed_time': res.get('elapsed_time', 0),
                        'status': res.get('status', 'unknown')
                    })
                    
                    # Вычисляем композитную оценку
                    composite = self.calculate_composite_score(res['test_metrics'], res['params'])
                    row.update(composite)
                    
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_csv(self.results_file, index=False)
                print(f"✓ Результаты сохранены в {self.results_file}")
        
        # Сохраняем сводку
        summary = {
            'pair_name': self.pair_name,
            'instrument1': self.instrument1,
            'instrument2': self.instrument2,
            'total_experiments': len(all_results),
            'successful': len(successful),
            'failed': len(failed),
            'search_completed': datetime.now().isoformat(),
            'best_models': self.get_best_models_summary(successful)
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Сводка сохранена в {self.summary_file}")
    
    def get_best_models_summary(self, successful_results):
        """Получение сводки по лучшим моделям"""
        if not successful_results:
            return {}
        
        # Создаем DataFrame из результатов
        df_data = []
        for res in successful_results:
            if 'test_metrics' in res:
                row = res['params'].copy()
                row.update({
                    'exp_id': res['exp_id'],
                    'best_test_loss': res.get('best_test_loss', 0),
                    'psnr_mean': res['test_metrics'].get('psnr_mean', 0),
                    'smoothness_score': res['test_metrics'].get('smoothness_score', 0),
                })
                
                composite = self.calculate_composite_score(res['test_metrics'], res['params'])
                row['composite_score'] = composite['composite_score']
                
                df_data.append(row)
        
        if not df_data:
            return {}
        
        df = pd.DataFrame(df_data)
        
        summary = {}
        
        # Лучшие по каждой метрике
        metrics_to_check = [
            ('composite_score', 'max'),
            ('psnr_mean', 'max'),
            ('smoothness_score', 'min'),  # Меньше = лучше
            ('best_test_loss', 'min')     # Меньше = лучше
        ]
        
        for metric, optimization in metrics_to_check:
            if metric in df.columns:
                try:
                    if optimization == 'max':
                        best_idx = df[metric].idxmax()
                    else:
                        best_idx = df[metric].idxmin()
                    
                    best_row = df.loc[best_idx]
                    summary[f'best_{metric}'] = {
                        'value': float(best_row[metric]),
                        'exp_id': int(best_row['exp_id']),
                        'latent_dim': int(best_row['latent_dim']),
                        'beta': float(best_row['beta']),
                        'learning_rate': float(best_row['learning_rate']),
                        'batch_size': int(best_row['batch_size'])
                    }
                except:
                    pass
        
        return summary
    
    def analyze_results(self, all_results):
        """Анализ результатов поиска"""
        print(f"\nАНАЛИЗ РЕЗУЛЬТАТОВ:")
        print("-" * 60)
        
        # Загружаем результаты если есть файл
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            
            if len(df) > 0:
                # Лучшая модель по композитной оценке
                if 'composite_score' in df.columns:
                    best_composite = df.loc[df['composite_score'].idxmax()]
                    print(f"Лучшая модель (композитная оценка):")
                    print(f"  Оценка: {best_composite['composite_score']:.4f}")
                    print(f"  PSNR: {best_composite['psnr_mean']:.2f} dB")
                    print(f"  Плавность: {best_composite['smoothness_score']:.4f}")
                    print(f"  Параметры:")
                    print(f"    latent_dim: {best_composite['latent_dim']}")
                    print(f"    beta: {best_composite['beta']}")
                    print(f"    learning_rate: {best_composite['learning_rate']}")
                    print(f"    batch_size: {best_composite['batch_size']}")
                    print(f"  ID эксперимента: {best_composite['exp_id']}")
                
                # Корреляции параметров с результатами
                print(f"\nКорреляции параметров с композитной оценкой:")
                for param in ['latent_dim', 'beta', 'learning_rate', 'batch_size']:
                    if param in df.columns and 'composite_score' in df.columns:
                        correlation = df[param].corr(df['composite_score'])
                        print(f"  {param}: {correlation:.3f}")
        
        # Создаем визуализации
        self.create_visualizations()
    
    def create_visualizations(self):
        """Создание визуализаций результатов поиска"""
        if not self.results_file.exists():
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.read_csv(self.results_file)
            
            if len(df) < 2:
                return
            
            # Настройки стиля
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # 1. Heatmap: beta vs latent_dim
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            if all(col in df.columns for col in ['beta', 'latent_dim', 'composite_score']):
                pivot_table = df.pivot_table(
                    values='composite_score',
                    index='latent_dim',
                    columns='beta',
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd',
                           ax=axes[0, 0], cbar_kws={'label': 'Composite Score'})
                axes[0, 0].set_title('Composite Score\n(beta vs latent_dim)')
                axes[0, 0].set_xlabel('Beta')
                axes[0, 0].set_ylabel('Latent Dimension')
            
            # 2. Scatter: PSNR vs Smoothness
            if all(col in df.columns for col in ['psnr_mean', 'smoothness_score']):
                scatter = axes[0, 1].scatter(
                    df['psnr_mean'],
                    df['smoothness_score'],
                    c=df['composite_score'],
                    s=df['latent_dim']*5,
                    alpha=0.6,
                    cmap='viridis'
                )
                
                axes[0, 1].set_xlabel('PSNR (dB)')
                axes[0, 1].set_ylabel('Smoothness Score')
                axes[0, 1].set_title('Качество реконструкции vs Плавность')
                axes[0, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 1], label='Composite Score')
            
            # 3. Влияние learning_rate на композитную оценку
            if 'learning_rate' in df.columns:
                axes[1, 0].scatter(df['learning_rate'], df['composite_score'], alpha=0.6)
                axes[1, 0].set_xlabel('Learning Rate')
                axes[1, 0].set_ylabel('Composite Score')
                axes[1, 0].set_title('Влияние Learning Rate')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Влияние batch_size на композитную оценку
            if 'batch_size' in df.columns:
                axes[1, 1].scatter(df['batch_size'], df['composite_score'], alpha=0.6)
                axes[1, 1].set_xlabel('Batch Size')
                axes[1, 1].set_ylabel('Composite Score')
                axes[1, 1].set_title('Влияние Batch Size')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Гиперпараметрический поиск: {self.instrument1} + {self.instrument2}', 
                        fontsize=16)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'hyperparameter_analysis.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Визуализации сохранены в {self.results_dir}")
            
        except Exception as e:
            print(f"⚠️  Ошибка при создании визуализаций: {e}")

def main():
    parser = argparse.ArgumentParser(description='Гиперпараметрический поиск для пары инструментов')
    parser.add_argument('--config', type=str, required=True,
                       help='Конфигурационный файл поиска')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для вычислений')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Максимальное количество экспериментов')
    
    args = parser.parse_args()
    
    # Проверяем конфигурационный файл
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Конфигурационный файл не найден: {config_path}")
        print("Создаю базовый конфиг...")
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        base_config = """hyperparam_pair_search:
  results_dir: "hyperparam_pair_results"
  
  instrument_pair:
    name: "flute_vocal"
    instrument1: "flute"
    instrument2: "vocal"
  
  param_grid:
    latent_dim: [16, 32, 64]
    beta: [0.1, 0.5, 1.0, 2.0]
    learning_rate: [0.001, 0.0005]
    batch_size: [16, 32]
  
  training:
    epochs: 100
    early_stopping_patience: 20
    train_ratio: 0.8
    num_workers: 2
  
  metrics:
    reconstruction_weight: 0.6
    smoothness_weight: 0.4
"""
        with open(config_path, 'w') as f:
            f.write(base_config)
        
        print(f"✓ Создан конфиг: {config_path}")
        print("При необходимости отредактируйте файл и запустите скрипт снова.")
        return
    
    # Проверяем данные
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Ошибка: директория с данными не найдена: {data_dir}")
        return
    
    # Запускаем поиск
    searcher = PairHyperparameterSearch(args.config)
    results = searcher.run_search(
        data_dir=str(data_dir),
        device=args.device,
        max_experiments=args.max_experiments
    )
    
    print("\n" + "=" * 70)
    print("ГИПЕРПАРАМЕТРИЧЕСКИЙ ПОИСК ЗАВЕРШЕН!")
    print("=" * 70)
    print(f"Результаты сохранены в: {searcher.results_dir}")

if __name__ == "__main__":
    main()