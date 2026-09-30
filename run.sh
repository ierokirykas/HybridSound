# Скрипт для запуска улучшенного пайплайна
# run_improved_pipeline.sh

#!/bin/bash

echo "================================================================"
echo "УЛУЧШЕННЫЙ ПАЙПЛАЙН: БОРЬБА С АРТЕФАКТАМИ АТАКИ"
echo "================================================================"

echo "1. Проверка данных..."
if [ ! -d "data/processed/nsynth_train_processed" ]; then
    echo "Ошибка: данные не найдены"
    exit 1
fi

echo "2. Обучение улучшенной VAE модели..."
python scripts/train_improved_vae.py \
  --data_dir "data/processed/nsynth_train_processed" \
  --checkpoint_dir "checkpoints/improved_vae" \
  --batch_size 32 \
  --latent_dim 64 \
  --attack_latent_dim 16 \
  --beta 0.1 \
  --epochs 150 \
  --lr 0.0005 \
  --device "cuda"

echo "3. Тестирование гибридизации с постобработкой..."
python scripts/test_hybridization.py \
  --checkpoint "checkpoints/final_vae_150epochs/best_model.pth" \
  --data_dir "data/processed/nsynth_train_processed" \
  --output_dir "hybridization_results/improved" \
  --device "cuda"

echo "4. Применение постобработки к результатам..."
python -c "
from src.audio.postprocessor import AudioPostProcessor
import glob
import os

processor = AudioPostProcessor(sample_rate=16000)
audio_files = glob.glob('hybridization_results/improved/audio_demos/*/*.wav')

print(f'Найдено {len(audio_files)} аудиофайлов для обработки...')

for audio_file in audio_files:
    output_dir = os.path.join('hybridization_results/improved/postprocessed', 
                             os.path.dirname(audio_file).split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = processor.batch_process_files(
        [audio_file],
        output_dir,
        fade_in=True,
        click_removal=True,
        compression=True,
        reverb=True,
        spectral_smoothing=True
    )
    
    print(f'Обработан: {audio_file}')
"

echo "================================================================"
echo "ПАЙПЛАЙН ЗАВЕРШЕН!"
echo "================================================================"
echo "Результаты с постобработкой в: hybridization_results/improved/postprocessed/"