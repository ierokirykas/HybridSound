#!/bin/bash
# run_complete_pipeline.sh

echo "================================================================"
echo "ПОЛНЫЙ ПАЙПЛАЙН: ОТ ДАННЫХ ДО ГИБРИДИЗАЦИИ"
echo "================================================================"

echo "1. Проверка данных..."
if [ ! -d "data/processed/nsynth_train_processed" ]; then
    echo "Данные не найдены. Запуск пайплайна обработки..."
    python scripts/run_pipeline.py --full
fi

echo "2. Анализ лучшей модели из гиперпараметрического поиска..."
python scripts/analyze_vae.py \
  --checkpoint "hyperparam_search_results/exp_016/best_model.pth" \
  --data_dir "data/processed/nsynth_train_processed" \
  --output_dir "analysis_results/best_model_analysis" \
  --batch_size 32 \
  --device "cuda"

echo "3. Обучение финальной модели на 150 эпох..."
python scripts/train_final_model.py \
  --data_dir "data/processed/nsynth_train_processed" \
  --checkpoint_dir "checkpoints/final_vae_150epochs" \
  --device "cuda"

echo "4. Тестирование гибридизации..."
python scripts/test_hybridization.py \
  --checkpoint "hyperparam_search_results/exp_016/best_model.pth" \
  --data_dir "data/processed/nsynth_train_processed" \
  --output_dir "hybridization_results/final" \
  --device "cuda"

echo "================================================================"
echo "ПАЙПЛАЙН ЗАВЕРШЕН!"
echo "================================================================"
echo "Результаты:"
echo "  - Анализ модели: analysis_results/best_model_analysis/"
echo "  - Финальная модель: checkpoints/final_vae_150epochs/"
echo "  - Гибридизация: hybridization_results/final/"