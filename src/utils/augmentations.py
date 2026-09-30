# src/utils/augmentations.py
import torch
import torch.nn.functional as F
import numpy as np

class SpectrogramAugmenter:
    """Аугментации для спектрограмм"""
    
    def __init__(self, 
                 freq_mask_max=10,
                 time_mask_max=10,
                 freq_mask_prob=0.3,
                 time_mask_prob=0.3,
                 time_shift_max=5,
                 time_shift_prob=0.4):
        
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.time_shift_max = time_shift_max
        self.time_shift_prob = time_shift_prob
    
    def __call__(self, spectrograms):
        """Применение аугментаций к батчу спектрограмм"""
        batch_size, channels, n_mels, n_time = spectrograms.shape
        
        augmented = spectrograms.clone()
        
        for i in range(batch_size):
            # Frequency masking
            if np.random.random() < self.freq_mask_prob:
                f = np.random.randint(0, self.freq_mask_max)
                f0 = np.random.randint(0, n_mels - f)
                augmented[i, :, f0:f0+f, :] = 0
            
            # Time masking
            if np.random.random() < self.time_mask_prob:
                t = np.random.randint(0, self.time_mask_max)
                t0 = np.random.randint(0, n_time - t)
                augmented[i, :, :, t0:t0+t] = 0
            
            # Time shifting (circular shift)
            if np.random.random() < self.time_shift_prob:
                shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
                if shift != 0:
                    augmented[i] = torch.roll(augmented[i], shifts=shift, dims=-1)
        
        return augmented