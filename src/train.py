import hydra
import wandb
import os
from omegaconf import DictConfig, OmegaConf
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
