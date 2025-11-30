import os
# Set environment variables before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF info/warnings/errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress OneDNN message
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1' # Suppress Albumentations update check
os.environ['WANDB_MODE'] = 'offline' # Automate WandB (no prompt)
os.environ['WANDB_SILENT'] = 'true' # Suppress WandB logs

import hydra
import wandb
import os
from omegaconf import DictConfig, OmegaConf
import keras

# Enable Mixed Precision
keras.mixed_precision.set_global_policy('mixed_float16')

# Enable XLA
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

from src.training.trainer import Trainer
from src.utils.seeding import seed_everything

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Seed
    seed_everything(cfg.seed)
    
    # WandB
    if not cfg.debug:
        wandb.init(
            project="aptos-2019",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.backbone}_{cfg.data.image_size}"
        )
    
    # Train
    trainer = Trainer(cfg)
    trainer.train()
    
    if not cfg.debug:
        wandb.finish()

if __name__ == "__main__":
    main()
