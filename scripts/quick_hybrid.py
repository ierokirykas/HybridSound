# scripts/quick_hybrid.py
#!/usr/bin/env python3
"""
Быстрое создание гибрида без интерактивного режима.
Пример использования: python quick_hybrid.py --inst1 3 --inst2 7 --alpha 0.5
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.interactive_hybridizer import InteractiveHybridizer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Быстрое создание гибрида инструментов')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/final_vae_150epochs/best_model.pth',
                       help='Путь к чекпоинту модели')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--inst1', type=int, required=True,
                       help='Номер первого инструмента (1-11)')
    parser.add_argument('--inst2', type=int, required=True,
                       help='Номер второго инструмента (1-11)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Коэффициент смешивания (0.0-1.0)')
    parser.add_argument('--alphas', type=str,
                       help='Несколько коэффициентов через запятую (0.0,0.25,0.5,0.75,1.0)')
    parser.add_argument('--interpolation', type=str, default='linear',
                       choices=['linear', 'spherical'],
                       help='Метод интерполяции')
    parser.add_argument('--generate_audio', action='store_true',
                       help='Сгенерировать аудиофайлы')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для вычислений')
    
    args = parser.parse_args()
    
    # Создаем гибридизатор
    hybridizer = InteractiveHybridizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Выбираем инструменты
    if not hybridizer.select_instruments(args.inst1, args.inst2):
        return
    
    # Устанавливаем параметры
    if args.alphas:
        alphas = [float(a.strip()) for a in args.alphas.split(',')]
        hybridizer.current_selection['alpha'] = alphas
    else:
        hybridizer.current_selection['alpha'] = args.alpha
    
    hybridizer.current_selection['interpolation'] = args.interpolation
    
    # Показываем текущий выбор
    hybridizer.show_current_selection()
    
    # Генерируем гибрид
    hybridizer.generate_hybrid()
    
    # Генерируем аудио если нужно
    if args.generate_audio:
        # Находим последнюю созданную директорию
        output_dirs = list(hybridizer.output_base_dir.glob("*"))
        if output_dirs:
            latest_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
            
            # Пересоздаем спектрограммы для генерации аудио
            # (в реальном коде нужно было бы сохранить generated_specs)
            print(f"\nГенерация аудио в директории: {latest_dir}")
            # Здесь нужно добавить логику для генерации аудио

if __name__ == "__main__":
    main()