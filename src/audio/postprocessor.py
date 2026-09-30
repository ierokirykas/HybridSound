# src/audio/postprocessor.py
"""
Модуль для постобработки сгенерированных аудио, устранения артефактов атаки
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import librosa
import warnings
warnings.filterwarnings('ignore')

class AudioPostProcessor:
    """Класс для постобработки аудио, устранения щелчков и артефактов атаки"""
    
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def adaptive_fade_in(self, audio, fade_duration_ms=50, detection_threshold=0.1):
        """
        Адаптивный fade-in с обнаружением реального начала сигнала
        
        Args:
            audio: аудиосигнал (моно)
            fade_duration_ms: длительность fade в миллисекундах
            detection_threshold: порог для обнаружения начала сигнала
        
        Returns:
            Обработанный аудиосигнал
        """
        if len(audio.shape) > 1:
            # Берем моно
            audio = librosa.to_mono(audio)
        
        # Находим реальное начало сигнала (не тишины)
        start_idx = self._find_signal_start(audio, threshold=detection_threshold)
        
        # Длительность fade в сэмплах
        fade_samples = int(fade_duration_ms * self.sr / 1000)
        
        # Применяем fade-in
        result = audio.copy()
        
        # Если начало сигнала найдено, начинаем fade оттуда
        if start_idx > 0 and start_idx < len(audio) - fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            result[start_idx:start_idx+fade_samples] *= fade_in
            
            # Обнуляем всё до начала сигнала (тишина)
            result[:start_idx] = 0
        else:
            # Иначе применяем fade с начала
            fade_in = np.linspace(0, 1, min(fade_samples, len(audio)))
            result[:len(fade_in)] *= fade_in
        
        return result
    
    def _find_signal_start(self, audio, threshold=0.01, window_ms=10):
        """Находит начало сигнала по энергии"""
        window_samples = int(window_ms * self.sr / 1000)
        if window_samples < 1:
            window_samples = 1
        
        # Скользящее окно энергии
        energy = np.convolve(np.abs(audio), np.ones(window_samples)/window_samples, mode='same')
        
        # Нормализуем энергию
        if energy.max() > 0:
            energy = energy / energy.max()
        
        # Находим первый сэмпл выше порога
        for i in range(len(energy)):
            if energy[i] > threshold:
                # Возвращаем с небольшим запасом назад
                return max(0, i - window_samples//2)
        
        return 0
    
    def remove_clicks(self, audio, threshold=0.02, window_ms=5):
        """
        Удаляет щелчки (резкие скачки амплитуды)
        
        Args:
            audio: аудиосигнал
            threshold: порог для обнаружения щелчков (производная)
            window_ms: размер окна для сглаживания
        
        Returns:
            Аудио без щелчков
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        result = audio.copy()
        
        # Вычисляем производную (разность)
        derivative = np.abs(np.diff(result, prepend=result[0]))
        
        # Нормализуем производную
        if derivative.max() > 0:
            derivative = derivative / derivative.max()
        
        # Находим щелчки (резкие скачки)
        click_indices = np.where(derivative > threshold)[0]
        
        if len(click_indices) > 0:
            window_samples = int(window_ms * self.sr / 1000)
            if window_samples < 1:
                window_samples = 1
            
            # Сглаживаем щелчки
            for idx in click_indices:
                start = max(0, idx - window_samples)
                end = min(len(result), idx + window_samples + 1)
                
                if end - start > 3:
                    # Интерполяция вокруг щелчка
                    x = np.arange(start, end)
                    if start > 0 and end < len(result):
                        # Линейная интерполяция между значениями до и после щелчка
                        y_start = result[start-1]
                        y_end = result[end]
                        result[start:end] = np.linspace(y_start, y_end, end-start)
        
        return result
    
    def dynamic_range_compression(self, audio, threshold_db=-20, ratio=4, attack_ms=10, release_ms=100):
        """
        Простой компрессор для контроля динамического диапазона
        
        Args:
            audio: аудиосигнал
            threshold_db: порог срабатывания в dB
            ratio: коэффициент компрессии (4:1 и т.д.)
            attack_ms: время атаки компрессора в мс
            release_ms: время восстановления в мс
        
        Returns:
            Сжатый аудиосигнал
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Преобразуем в dB
        eps = 1e-8
        audio_db = 20 * np.log10(np.abs(audio) + eps)
        
        # Параметры в сэмплах
        attack_samples = int(attack_ms * self.sr / 1000)
        release_samples = int(release_ms * self.sr / 1000)
        
        # Применяем компрессию
        gain_reduction = np.zeros_like(audio_db)
        
        for i in range(len(audio_db)):
            if audio_db[i] > threshold_db:
                # Вычисляем необходимое ослабление
                excess_db = audio_db[i] - threshold_db
                reduction_db = excess_db * (1 - 1/ratio)
                target_reduction = reduction_db
            else:
                target_reduction = 0
            
            # Применяем атаку/восстановление
            if i == 0:
                gain_reduction[i] = target_reduction
            else:
                if target_reduction > gain_reduction[i-1]:
                    # Атака
                    gain_reduction[i] = gain_reduction[i-1] + (target_reduction - gain_reduction[i-1]) / attack_samples
                else:
                    # Восстановление
                    gain_reduction[i] = gain_reduction[i-1] - (gain_reduction[i-1] - target_reduction) / release_samples
        
        # Преобразуем ослабление в линейный множитель
        gain_linear = 10 ** (-gain_reduction / 20)
        
        # Применяем gain
        result = audio * gain_linear
        
        # Нормализуем
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return result
    
    def add_natural_reverb(self, audio, room_size=0.3, damping=0.5, wet_level=0.1):
        """
        Добавляет легкую реверберацию для естественности
        
        Args:
            audio: аудиосигнал
            room_size: размер комнаты (0-1)
            damping: демпфирование высоких частот (0-1)
            wet_level: уровень эффекта (0-1)
        
        Returns:
            Аудио с реверберацией
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Простая реверберация через FIR фильтр
        # Генерируем импульсную характеристику комнаты
        length = int(0.5 * self.sr)  # 0.5 секунды
        t = np.arange(length) / self.sr
        
        # Экспоненциально затухающий синусоидальный импульс
        ir = np.exp(-t * 3 / room_size) * np.sin(2 * np.pi * 1000 * t)
        
        # Применяем демпфирование высоких частот
        ir *= np.exp(-t * damping * 10)
        
        # Нормализуем
        ir = ir / np.sqrt(np.sum(ir**2))
        
        # Свертка (реверберация)
        wet = np.convolve(audio, ir, mode='same')
        
        # Смешиваем dry и wet сигналы
        result = (1 - wet_level) * audio + wet_level * wet
        
        return result
    
    def spectral_smoothing(self, audio, smoothing_factor=0.1):
        """
        Спектральное сглаживание для устранения артефактов
        
        Args:
            audio: аудиосигнал
            smoothing_factor: степень сглаживания (0-1)
        
        Returns:
            Сглаженный аудиосигнал
        """
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # STFT
        n_fft = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Логарифмическое сглаживание амплитуд
        log_mag = np.log1p(magnitude)
        smoothed_log_mag = gaussian_filter1d(log_mag, sigma=smoothing_factor*10, axis=0)
        smoothed_mag = np.expm1(smoothed_log_mag)
        
        # Восстанавливаем
        smoothed_stft = smoothed_mag * np.exp(1j * phase)
        result = librosa.istft(smoothed_stft, hop_length=hop_length)
        
        # Обрезаем до оригинальной длины
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))
        
        return result
    
    def full_postprocessing_pipeline(self, audio, 
                                    fade_in=True, 
                                    click_removal=True,
                                    compression=True,
                                    reverb=False,
                                    spectral_smoothing=False):
        """
        Полный пайплайн постобработки
        
        Args:
            audio: входной аудиосигнал
            fade_in: применять адаптивный fade-in
            click_removal: удалять щелчки
            compression: применять компрессию
            reverb: добавлять легкую реверберацию
            spectral_smoothing: применять спектральное сглаживание
        
        Returns:
            Обработанный аудиосигнал
        """
        result = audio.copy()
        
        if len(result.shape) > 1:
            result = librosa.to_mono(result)
        
        # 1. Удаление щелчков
        if click_removal:
            result = self.remove_clicks(result)
        
        # 2. Адаптивный fade-in
        if fade_in:
            result = self.adaptive_fade_in(result)
        
        # 3. Спектральное сглаживание
        if spectral_smoothing:
            result = self.spectral_smoothing(result, smoothing_factor=0.05)
        
        # 4. Компрессия
        if compression:
            result = self.dynamic_range_compression(result)
        
        # 5. Реверберация
        if reverb:
            result = self.add_natural_reverb(result, wet_level=0.08)
        
        # Финальная нормализация
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result)) * 0.9
        
        return result
    
    def batch_process_files(self, input_files, output_dir, 
                           sample_rate=16000, **kwargs):
        """
        Пакетная обработка аудиофайлов
        
        Args:
            input_files: список путей к входным файлам
            output_dir: директория для сохранения результатов
            sample_rate: частота дискретизации
            **kwargs: параметры для full_postprocessing_pipeline
        
        Returns:
            Список путей к обработанным файлам
        """
        import os
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        
        for input_file in input_files:
            try:
                # Загружаем аудио
                audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
                
                # Обрабатываем
                processed = self.full_postprocessing_pipeline(audio, **kwargs)
                
                # Сохраняем
                output_file = output_dir / f"processed_{Path(input_file).name}"
                sf.write(output_file, processed, sr)
                
                processed_files.append(str(output_file))
                
                print(f"✓ Обработан: {Path(input_file).name}")
                
            except Exception as e:
                print(f"✗ Ошибка при обработке {input_file}: {e}")
        
        return processed_files