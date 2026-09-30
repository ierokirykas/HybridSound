# scripts/train_improved_vae.py
"""
Обучение улучшенной VAE модели с учетом атаки
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class ImprovedVAE(nn.Module):
    """Улучшенная VAE с отдельным кодированием атаки и сустейна"""
    
    def __init__(self, 
                 input_shape=(1, 128, 63),
                 latent_dim=64,
                 attack_latent_dim=16,
                 beta=0.1,
                 attribute_dims=None):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.attack_latent_dim = attack_latent_dim
        self.beta = beta
        
        if attribute_dims is None:
            self.attribute_dims = {
                'instrument_family': 11,
                'instrument_source': 3,
                'attack_strength': 1  # Регрессия, не классификация
            }
        else:
            self.attribute_dims = attribute_dims
        
        # Основной энкодер (весь сигнал)
        self.main_encoder = self._build_encoder()
        
        # Энкодер атаки (первые N кадров)
        self.attack_encoder = self._build_attack_encoder()
        
        # Линейные слои для латентных переменных
        encoder_output_size = 256 * 8 * 4
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, latent_dim)
        
        # Отдельно для атаки
        attack_encoder_size = 128 * 8 * 2
        self.fc_attack_mu = nn.Linear(attack_encoder_size, attack_latent_dim)
        self.fc_attack_logvar = nn.Linear(attack_encoder_size, attack_latent_dim)
        
        # Декодер
        self.decoder_input = nn.Linear(latent_dim + attack_latent_dim, encoder_output_size)
        self.decoder = self._build_decoder()
        
        # Multi-task heads
        self.attribute_heads = nn.ModuleDict()
        for attr_name, num_classes in self.attribute_dims.items():
            if num_classes == 1:
                # Регрессия для attack_strength
                self.attribute_heads[attr_name] = nn.Sequential(
                    nn.Linear(latent_dim + attack_latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            else:
                # Классификация
                self.attribute_heads[attr_name] = nn.Sequential(
                    nn.Linear(latent_dim + attack_latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
    
    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
    
    def _build_attack_encoder(self):
        """Энкодер для первых 20 кадров (атака)"""
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
    
    def _build_decoder(self):
        return nn.Sequential(
            nn.Unflatten(1, (256, 8, 4)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Кодирование всего сигнала"""
        # Основной энкодер
        features = self.main_encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def encode_attack(self, x):
        """Кодирование только атаки (первые 20 временных кадров)"""
        attack_segment = x[:, :, :, :20]  # (batch, 1, n_mels, 20)
        features = self.attack_encoder(attack_segment)
        mu = self.fc_attack_mu(features)
        logvar = self.fc_attack_logvar(features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z_main, z_attack):
        """Декодирование из объединенного латентного пространства"""
        z_combined = torch.cat([z_main, z_attack], dim=1)
        x = self.decoder_input(z_combined)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        # Кодирование
        mu_main, logvar_main = self.encode(x)
        mu_attack, logvar_attack = self.encode_attack(x)
        
        # Репараметризация
        z_main = self.reparameterize(mu_main, logvar_main)
        z_attack = self.reparameterize(mu_attack, logvar_attack)
        
        # Декодирование
        x_recon = self.decode(z_main, z_attack)
        
        # Предсказание атрибутов
        z_combined = torch.cat([z_main, z_attack], dim=1)
        attr_preds = {}
        for attr_name, head in self.attribute_heads.items():
            attr_preds[attr_name] = head(z_combined)
        
        return x_recon, mu_main, logvar_main, mu_attack, logvar_attack, attr_preds
    
    def loss_function(self, x, x_recon, mu_main, logvar_main, mu_attack, logvar_attack, 
                     attr_preds, attr_targets, recon_weight=1.0):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL losses
        kl_main = -0.5 * torch.sum(1 + logvar_main - mu_main.pow(2) - logvar_main.exp()) / x.size(0)
        kl_attack = -0.5 * torch.sum(1 + logvar_attack - mu_attack.pow(2) - logvar_attack.exp()) / x.size(0)
        
        # Attribute losses
        attr_losses = {}
        total_attr_loss = 0
        attr_weights = {
            'instrument_family': 0.5,
            'instrument_source': 0.1,
            'attack_strength': 0.2
        }
        
        for attr_name, pred in attr_preds.items():
            if attr_name in attr_targets:
                target = attr_targets[attr_name]
                weight = attr_weights.get(attr_name, 0.1)
                
                if self.attribute_dims[attr_name] == 1:
                    # Регрессия (MSE)
                    loss = F.mse_loss(pred.squeeze(), target.float())
                else:
                    # Классификация (CrossEntropy)
                    loss = F.cross_entropy(pred, target)
                
                attr_losses[attr_name] = loss
                total_attr_loss += weight * loss
        
        # Total loss
        total_loss = (recon_weight * recon_loss + 
                     self.beta * (kl_main + kl_attack * 0.5) +  # Меньший вес для атаки
                     total_attr_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_main': kl_main,
            'kl_attack': kl_attack,
            'total_attr_loss': total_attr_loss,
            **{f'{name}_loss': loss for name, loss in attr_losses.items()}
        }

def main():
    parser = argparse.ArgumentParser(description='Обучение улучшенной VAE')
    parser.add_argument('--data_dir', type=str, 
                       default='data/processed/nsynth_train_processed',
                       help='Директория с данными')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints/improved_vae',
                       help='Директория для чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--attack_latent_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Импорт здесь чтобы избежать циклических импортов
    from src.data.improved_dataloader import create_improved_dataloader
    
    print("=" * 70)
    print("ОБУЧЕНИЕ УЛУЧШЕННОЙ VAE (С УЧЕТОМ АТАКИ)")
    print("=" * 70)
    
    # Создаем DataLoader
    data_dir = Path(args.data_dir)
    train_loader, val_loader, dataset = create_improved_dataloader(
        spectrograms_path=str(data_dir / "melspectrograms.npy"),
        metadata_path=str(data_dir / "processed_metadata.json"),
        batch_size=args.batch_size,
        attack_threshold=0.3,
        offset_ms=50
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Создаем модель
    model = ImprovedVAE(
        latent_dim=args.latent_dim,
        attack_latent_dim=args.attack_latent_dim,
        beta=args.beta,
        attribute_dims={
            'instrument_family': 11,
            'instrument_source': 3,
            'attack_strength': 1
        }
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    print("\nНачало обучения...")
    
    # Тренировочный цикл (упрощенный)
    # Здесь должен быть полный цикл обучения как в train_vae.py
    
    print("Для полного обучения используйте адаптированный скрипт обучения")
    
    # Сохраняем модель
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'latent_dim': args.latent_dim,
            'attack_latent_dim': args.attack_latent_dim,
            'beta': args.beta
        }
    }, checkpoint_dir / "model_init.pth")
    
    print(f"\nМодель инициализирована и сохранена в {checkpoint_dir}")

if __name__ == "__main__":
    main()