# src/models/fixed_vae.py - ИСПРАВЛЕННЫЙ КОД

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class FixedSizeVAE(nn.Module):
    """VAE с фиксированными размерами для спектрограмм 128x63 - УПРОЩЕННАЯ ВЕРСИЯ"""
    
    def __init__(self, 
                 latent_dim: int = 16,  # Уменьшено с 64
                 attribute_dims: Dict[str, int] = None,
                 beta: float = 10.0,  # Увеличено с 1.0
                 attr_weights: Dict[str, float] = None):
        """
        Args:
            latent_dim: размерность латентного пространства
            attribute_dims: словарь с размерностями атрибутов {name: num_classes}
            beta: вес KL-дивергенции (β в β-VAE) - УВЕЛИЧЕН
            attr_weights: веса для loss каждого атрибута
        """
        super().__init__()
        
        self.input_shape = (1, 128, 63)  # Фиксированный размер
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Атрибуты по умолчанию
        if attribute_dims is None:
            self.attribute_dims = {
                'instrument_family': 11,
                'instrument_source': 3,
                'brightness': 3,
                'attack_type': 1
            }
        else:
            self.attribute_dims = attribute_dims
        
        # Веса атрибутов
        if attr_weights is None:
            self.attr_weights = {name: 0.1 for name in self.attribute_dims}
        else:
            self.attr_weights = attr_weights
        
        # Энкодер - УПРОЩЕН
        self.encoder = nn.Sequential(
            # Блок 1: 128x63 -> 64x32
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 2: 64x32 -> 32x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 3: 32x16 -> 16x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 4: 16x8 -> 8x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Flatten()
        )
        
        # Размер выхода энкодера: 128 * 8 * 4 = 4096
        encoder_output_size = 128 * 8 * 4
        
        # Линейные слои для параметров распределения
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_var = nn.Linear(encoder_output_size, latent_dim)
        
        # Инициализация с малыми значениями
        nn.init.normal_(self.fc_mu.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.normal_(self.fc_var.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_var.bias, 0)
        
        # Декодер - УПРОЩЕН
        self.decoder_input = nn.Linear(latent_dim, encoder_output_size)
        
        self.decoder = nn.Sequential(
            # Восстановление формы: 128 * 8 * 4 -> (128, 8, 4)
            nn.Unflatten(1, (128, 8, 4)),
            
            # Блок 1: 8x4 -> 16x8
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 2: 16x8 -> 32x16
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 3: 32x16 -> 64x32
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Блок 4: 64x32 -> 128x63
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Финальный слой
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Multi-task heads - УПРОЩЕНЫ
        self.attribute_heads = nn.ModuleDict()
        for attr_name, num_classes in self.attribute_dims.items():
            self.attribute_heads[attr_name] = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes)
            )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кодирование в латентное пространство с ограничением дисперсии"""
        result = self.encoder(x)
        mu = self.fc_mu(result)
        
        # Ограничиваем log_var чтобы избежать взрыва KL
        log_var = self.fc_var(result)
        log_var = torch.clamp(log_var, min=-10, max=10)  # Защита от взрыва
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick для обучения VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Декодирование из латентного пространства"""
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result
    
    def predict_attributes(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Предсказание атрибутов из латентного вектора"""
        predictions = {}
        for attr_name, head in self.attribute_heads.items():
            predictions[attr_name] = head(z)
        return predictions
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Прямой проход через модель"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        # Реконструкция
        x_recon = self.decode(z)
        
        # Предсказание атрибутов
        attr_predictions = self.predict_attributes(z)
        
        return x_recon, mu, log_var, attr_predictions
    
    def loss_function(self, 
                     x: torch.Tensor,
                     x_recon: torch.Tensor,
                     mu: torch.Tensor,
                     log_var: torch.Tensor,
                     attr_predictions: Dict[str, torch.Tensor],
                     attr_targets: Dict[str, torch.Tensor],
                     recon_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """Вычисление функции потерь с защитой от численных проблем"""
        
        batch_size = x.size(0)
        
        # Reconstruction loss
        if x.shape != x_recon.shape:
            x_recon = F.interpolate(x_recon, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # KL divergence с защитой от NaN/Inf
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        
        # Проверка на численные проблемы
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            kl_loss = torch.tensor(0.0, device=x.device)
        
        # Потери классификации атрибутов
        attr_losses = {}
        total_attr_loss = 0.0
        
        for attr_name, pred in attr_predictions.items():
            if attr_name in attr_targets:
                target = attr_targets[attr_name]
                weight = self.attr_weights.get(attr_name, 0.1)
                
                # Cross-entropy loss
                attr_loss = F.cross_entropy(pred, target, reduction='mean')
                attr_losses[attr_name] = attr_loss
                total_attr_loss += weight * attr_loss
        
        # Общая потеря
        total_loss = recon_weight * recon_loss + self.beta * kl_loss + total_attr_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_attr_loss': total_attr_loss,
            **{f'{name}_loss': loss for name, loss in attr_losses.items()}
        }