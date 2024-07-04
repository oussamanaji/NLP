from dataclasses import dataclass

@dataclass
class BiasGuardConfig:
    model_id: str = "CohereForAI/aya-23-8B"
    num_epochs: int = 10
    prompts_per_epoch: int = 100
    eval_steps: int = 500
    output_dir: str = "./experiment_results"
    learning_rate: float = 2e-5
    batch_size: int = 64
    max_length: int = 100
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 5
    ppo_batch_size: int = 16
    ppo_lr: float = 1e-5
    ppo_clip_epsilon: float = 0.2
    ppo_value_loss_coef: float = 0.1
    ppo_entropy_coef: float = 0.01
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1

# Usage
if __name__ == "__main__":
    config = BiasGuardConfig()
    print(config)
