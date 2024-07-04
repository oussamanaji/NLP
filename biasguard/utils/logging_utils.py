import logging
import sys
import wandb

def setup_logging(log_file="biasguard.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def init_wandb(config):
    wandb.init(
        project="BiasGuard",
        config=config.__dict__
    )

def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
    for key, value in metrics.items():
        logging.info(f"{key}: {value}")

# Usage
if __name__ == "__main__":
    from utils.config import BiasGuardConfig

    setup_logging()
    config = BiasGuardConfig()
    init_wandb(config)
    
    # Example logging
    logging.info("Starting training...")
    log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
