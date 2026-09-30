# HybridSound — генерация гибридных тембров инструментов

Курсовая работа. Модель обучается латентному пространству тембров на датасете
[NSynth](https://magenta.tensorflow.org/datasets/nsynth), а затем интерполирует
латентные векторы двух инструментов и синтезирует звук с тембром «на стыке» —
например, нечто среднее между флейтой и вокалом.

Полный цикл: скачивание датасета → препроцессинг в мел-спектрограммы →
обучение β-VAE с multi-task классификационными головами → анализ латентного
пространства → гибридизация → постобработка аудио.

---

## Что делает

1. **Скачивает и сэмплирует NSynth** — 11 семейств инструментов, сбалансированная
   выборка ~25 000 нот из train-сплита (`scripts/download_nsynth_train.py`).
2. **Препроцессит аудио** — моно, ресемплинг, RMS-нормализация, центральный
   отрезок 2 с @ 16 кГц, log-mel спектрограмма `(128, 63)`, кодирование
   атрибутов → `melspectrograms.npy` + `processed_metadata.json`
   (`src/data_pipeline.py`).
3. **Обучает β-VAE** (`src/models/fixed_vae.py`) — свёрточный VAE над
   log-mel-спектрограммами фиксированного размера + 4 атрибутивные головы
   (`instrument_family`, `instrument_source`, `brightness`, `attack_type`).
4. **Анализирует** — MSE/PSNR реконструкции, t-SNE и PCA латентного
   пространства, точность по каждому атрибуту, матрица ошибок, 7-шаговая
   интерполяция (`scripts/analyze_vae.py`).
5. **Гибридизирует** — усредняет латент по семейству, интерполирует между
   двумя инструментами (линейно или сферически), декодирует, восстанавливает
   аудио (`scripts/interactive_hybridizer.py`, `scripts/test_hybridization.py`).
6. **Постобрабатывает** аудио — удаление кликов, адаптивный fade-in, спектральное
   сглаживание, компрессия динамического диапазона, реверберация
   (`src/audio/postprocessor.py`).

---

## Архитектура модели

`src/models/fixed_vae.py` — `FixedSizeVAE(latent_dim=16, beta=10.0, ...)`.
Вход жёстко привязан к форме `(1, 128, 63)` (128 мел-полос × 63 кадра).

```
Encoder                                     Decoder
─────────────────────────────────────       ─────────────────────────────────────
Conv2d  1 →  16  k3 s2 p1                   Linear(latent_dim → 4096)
Conv2d 16 →  32  k3 s2 p1                   Unflatten → (128, 8, 4)
Conv2d 32 →  64  k3 s2 p1                   ConvTranspose2d 128 →  64  k3 s2 p1
Conv2d 64 → 128  k3 s2 p1                   ConvTranspose2d  64 →  32  k3 s2 p1
                                           ConvTranspose2d  32 →  16  k3 s2 p1
                                           ConvTranspose2d  16 →   8  k3 s2 p1
Flatten → 4096                              Conv2d 8 → 1 k3 s1 p1 → Sigmoid
```

Каждый блок: `BatchNorm2d → ReLU → Dropout2d(0.1)`. Последний
`ConvTranspose2d` использует `output_padding=(1, 0)`, чтобы попасть ровно в
128×63.

Отдельно стоят две линейные головы `fc_mu` и `fc_var` (4096 → latent_dim),
инициализированные `N(0, 0.01)`. `log_var` ограничивается `clamp(-10, 10)` —
защита от взрыва KL.

**Атрибутивные головы** — по одной линейной ветке на атрибут из общего
латентного вектора:

```python
self.attribute_heads[attr] = nn.Sequential(
    nn.Linear(latent_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes)
)
```

**Функция потерь** — взвешенная сумма:

```
L = w_rec · MSE(x, x̂)  +  β · KL(q(z|x) ‖ p(z))  +  Σ_i w_i · CE_i(attr_i, ŷ_i)
```

`β = 10.0` по умолчанию. Веса атрибутов задаются в `scripts/train_vae.py:90-95`:
`{family: 0.5, source: 0.2, brightness: 0.05, attack: 0.01}`.

### Численная устойчивость

- `clamp(log_var, -10, 10)` — защита от вырожденной дисперсии;
- KL сбрасывается в ноль при `NaN`/`Inf` (`fixed_vae.py:199-200`);
- пропуск батча при `NaN` в `x_recon` / `mu` / `log_var` (`train_vae.py:163`);
- клиппинг градиентов (`max_grad_norm = 1.0`);
- проверка формы выхода **до** старта обучения (`train_vae.py`) — именно она
  поймала несовпадение размеров в ранних экспериментах.

---

## Результаты

Замеры — `analysis_results/`, валидационная выборка 999 сэмплов.

| Метрика | Базовая модель (`vae_test`, latent 16, β=10) | Лучшая модель (`exp_016_best`, latent 64, β=0.1) |
|---|---|---|
| PSNR | 18.11 dB | **23.70 dB** |
| MSE | 0.01782 | **0.00542** (×3.3) |
| best val loss | 179.40 | **54.95** (×3.3) |
| `instrument_family` (11 классов) | 23.4 % | **39.0 %** |
| `instrument_source` (3 класса) | 48.6 % | **66.8 %** |
| `brightness` (3 класса) | 56.1 % | **78.5 %** |
| `attack_type` (3 класса) | 42.4 % | 47.8 % |
| silhouette score | −0.093 | −0.034 |

### Честная интерпретация

**`silhouette_score` отрицательный** — латентное пространство *не* разбивается
на компактные кластеры по семействам инструментов. Про декомпозицию тембра на
ортогональные компоненты говорить нельзя.

Корректная формулировка результата: это multi-task β-VAE, который
реконструирует log-mel-спектрограммы (PSNR 23.7 dB) и чьи атрибутивные головы
достигают 39 % по семейству инструментов при 11 классах (случайное угадывание
дало бы 9.1 %) и 67 % по типу источника. Атрибутивные головы действительно
выучивают информацию о тембре из латента — это видно по тому, что реконструкция
сама по себе (β = 10) эту классификацию не обеспечивает.

### Известные ограничения

- В текущем `config/data_config.yaml` атрибуты `brightness` и `attack_type`
  вырождены: `brightness` отображает только `medium: [49, 72]`, `attack_type` —
  только `soft: [0, 64]`, и обе ветки имеют `default` на ту же единственную
  метку. Из-за этого текущий конфиг даёт по 1 классу вместо 3. Значения с
  3 классами в таблице выше относятся к более ранней версии датасета.
- **Фаза в вокодере случайная, а не Griffin-Lim.** Все три реализации
  `spec_to_audio` делают один розыгрыш `np.random.uniform(0, 2π)`, без
  итераций. Комментарии в коде называют это «упрощённым Griffin-Lim» — это
  неточно.
- Обратное преобразование мел-базиса не инвертируется (`analyze_vae.py`
  обрабатывает мел-полосы как bins STFT), а шкала dB восстанавливается как
  `spec * 80 - 40` по уже нормализованному значению. **Перцептивная точность
  звука не гарантирована.**
- Рендер аудио короче 2 с (~1 с) — это кроп результата `istft`, а не полный
  фрагмент.
- `pipeline.sh` и `run.sh` ссылаются на `hyperparam_search_results/exp_016/`,
  которого в репозитории нет. Смотрите раздел «Запуск» — там ручная
  последовательность, а не эти скрипты.
- Все 4 сохранённых прогона поиска гиперпараметров (`exp_001`…`exp_004`)
  завершились ошибкой. Успешного CSV/JSON-сводки по поиску нет. Ось
  `weight_family` в конфиге — no-op, потому что `train_vae.py` хардкодит
  `attr_weights`.
- Тестовое покрытие фактически отсутствует: `src/tests/test_data_pipeline.py`
  состоит из трёх функций с `pass`. Рабочий смоук-тест формы — `quick_fix.py`.

---

## Структура

```
config/                     7 YAML-конфигов: данные, датасет, 2 сетки поиска, 2 модели
src/
  data_pipeline.py          препроцессинг: валидация, загрузка, mel, атрибуты
  models/fixed_vae.py       FixedSizeVAE — основная модель
  data/dataset.py           AudioSpectrogramDataset + AudioDataLoader
  data/augmentations.py     SpecAugment-подобные аугментации с сохранением формы
  data/improved_dataloader.py   вариант с attack-aware кодированием (не в основном пути)
  training/trainer.py       VAETrainer с early stopping (не используется скриптами)
  audio/postprocessor.py    постобработка аудио
  tests/                    (заглушки)
scripts/
  download_nsynth_subset.py     сэмплирование test-сплита
  download_nsynth_train.py      сэмплирование train-сплита, 3 стратегии распаковки
  check_nsynth_status.py        статус выкачанных данных
  run_pipeline.py               драйвер препроцессинга (--test / --full)
  train_vae.py                  основной тренер
  train_pair_vae.py             специализированная VAE под пару инструментов
  train_improved_vae.py         двухэнкодерный вариант с attack-головой
  train_final_model.py          обёртка с зафиксированными лучшими гиперпараметрами
  hyperparameter_search.py      grid search 180 комбинаций
  hyperparameter_search_pair.py grid search 48 комбинаций (flute + vocal)
  analyze_vae.py                офлайн-оценка чекпойнта
  test_hybridization.py         попарная гибридизация, α ∈ {0, .25, .5, .75, 1}
  interactive_hybridizer.py    консольный REPL смешивания тембров
  quick_hybrid.py               неинтерактивная обёртка
  download_preprocessed.py      (заглушка, URL-заглушка)
```

Второй по важности артефакт — `analysis_results/`: t-SNE/PCA латентного
пространства, сетки реконструкций, полосы интерполяции, `analysis_summary.json`
с метриками.

---

## Запуск

Зависимостей ровно две: **PyTorch с CUDA** и **librosa**. `torchaudio` в коде
не используется — вся работа со звуком идёт через `librosa` + `soundfile`.

```bash
# 1. зависимости (Python 3.12)
pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu118
pip install numpy==2.3.3 scipy==1.16.3 librosa==0.11.0 soundfile==0.13.1 \
            pandas==2.3.3 scikit-learn==1.8.0 matplotlib==3.10.8 \
            seaborn==0.13.2 PyYAML==6.0.3 tqdm==4.67.1 tensorboard==2.20.0

# 2. данные
python scripts/download_nsynth_subset.py --config config/nsynth_config.yaml
python scripts/download_nsynth_train.py  --config config/nsynth_train_config.yaml
python scripts/check_nsynth_status.py

# 3. препроцессинг
python scripts/run_pipeline.py --test    # смоук на 5 файлах
python scripts/run_pipeline.py --full    # полный датасет

# 4. обучение
python scripts/train_vae.py \
    --data_dir data/processed/nsynth_train_processed \
    --batch_size 32 --latent_dim 64 --epochs 150 --lr 0.001 --beta 0.2 \
    --checkpoint_dir checkpoints/final_vae_150epochs --device cuda --grad_clip 1.0

# 5. анализ
python scripts/analyze_vae.py \
    --checkpoint checkpoints/final_vae_150epochs/best_model.pth \
    --data_dir data/processed/nsynth_train_processed \
    --output_dir analysis_results/final_vae --device cuda
tensorboard --logdir runs/

# 6. гибридизация
python scripts/test_hybridization.py \
    --checkpoint checkpoints/final_vae_150epochs/best_model.pth \
    --data_dir data/processed/nsynth_train_processed \
    --output_dir hybridization_results/final --device cuda

# 7. интерактивно
python scripts/interactive_hybridizer.py \
    --checkpoint checkpoints/final_vae_150epochs/best_model.pth \
    --data_dir data/processed/nsynth_train_processed --device cuda
```

Все команды запускаются **из корня репозитория**: пакетных `__init__.py` нет,
импорты работают через `sys.path.append(str(project_root))` в каждом скрипте.

Поиск гиперпараметров (долгий, 180 комбинаций × 20 эпох):

```bash
python scripts/hyperparameter_search.py \
    --data_dir data/processed/nsynth_train_processed \
    --config config/hyperparam_config.yaml --max_experiments 20 --visualize
```

> CUDA захардкожена почти во всех точках входа: `--device` по умолчанию
> `cuda`. `train_vae.py` и `analyze_vae.py` аккуратно откатываются на CPU, если
> CUDA недоступна; остальные скрипты — нет. Разработка и обучение велись на
> RTX 3050 (Windows 11).

---

## Стек

| Пакет | Версия |
|---|---|
| Python | 3.12 |
| torch | 2.7.1+cu118 |
| numpy | 2.3.3 |
| scipy | 1.16.3 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.1 |
| tensorboard | 2.20.0 |

Кода в репозитории ~7 700 строк Python в 29 файлах.

---

## Атрибуция

Датасет **NSynth**, Google Magenta, лицензия **CC BY 4.0**:

> Jesse Engel, Kumar Krishna Agrawal, Chenjie Gu, Adam Roberts, Zefeng Zhu,
> Douglas Eck. *Synthesizing Adjective Sound Descriptors* / NSynth dataset.

Аудиофайлы NSynth в репозиторий **не** включены — скрипты скачивают их сами.
Все `.wav` в `hybridization_results/` сгенерированы моделью.

---

## Что не коммитить

- `checkpoints/` — 317 МБ (55 файлов, из них 42 никогда не загружаются).
  Один `best_model.pth` (~12 МБ) — в Releases или через Git LFS.
- `images/` — 7 PNG со случайными GUID-именами, это скриншоты чата, а не
  результаты проекта.
- `hybridization_results/flute_reed.lnk` — ярлык Windows.
- `hybridization_results/` целиком — 37 МБ (467 файлов). В репозиторий стоит
  положить 3–5 пар, а не все.
- `__pycache__/`, включая два «сиротских» `.pyc` без исходников
  (`improved_vae`, `multitask_vae`).
- Устаревшие дубликаты: `scripts/train_vae copy.py`,
  `src/models/fixed_vae copy.py` (пробел в имени ломает упаковку).
- `runs/` — 43 event-файла TensorBoard, в каждом видно имя хоста
  `DESKTOP-6AI6FM9` и PID.

---

## Лицензия кода

MIT. Лицензия датасета — CC BY 4.0, см. «Атрибуция» выше.
