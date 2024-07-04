from trl import PPOTrainer, PPOConfig
from transformers import TrainingArguments
import torch

class BiasGuardPPOTrainer:
    def __init__(self, actor_model, critic_model, reward_model, tokenizer):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
        ppo_config = PPOConfig(
            model_name="CohereForAI/aya-23-8B",
            learning_rate=2e-5,
            batch_size=64,
            ppo_epochs=5,
            gradient_accumulation_steps=4,
            optimize_cuda_cache=True
        )
        
        training_args = TrainingArguments(
            output_dir="./ppo_results",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            warmup_steps=100,
            logging_steps=10,
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.actor_model,
            ref_model=None,
            tokenizer=self.tokenizer,
            dataset=None,  # We'll provide data during training
            data_collator=None,
            args=training_args
        )

    def train_step(self, prompts, max_length=100):
        # Generate responses
        responses = [self.actor_model.generate(prompt, max_length) for prompt in prompts]
        
        # Evaluate bias
        bias_scores = [self.critic_model.evaluate_bias(response) for response in responses]
        
        # Compute rewards
        rewards = [self.reward_model.compute_reward(response, bias_score) 
                   for response, bias_score in zip(responses, bias_scores)]
        
        # Prepare data for PPO
        query_tensors = [self.tokenizer.encode(prompt, return_tensors="pt").to(self.actor_model.device) for prompt in prompts]
        response_tensors = [self.tokenizer.encode(response, return_tensors="pt").to(self.actor_model.device) for response in responses]
        
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        return stats, responses, bias_scores, rewards

# Usage
if __name__ == "__main__":
    from models.actor_model import ActorModel
    from models.critic_model import CriticModel
    from models.reward_model import RewardModel

    actor = ActorModel()
    critic = CriticModel()
    reward_model = RewardModel()
    tokenizer = actor.tokenizer

    trainer = BiasGuardPPOTrainer(actor, critic, reward_model, tokenizer)
    prompts = ["Discuss the role of women in the workplace."]
    stats, responses, bias_scores, rewards = trainer.train_step(prompts)
    print(f"Response: {responses[0]}")
    print(f"Bias score: {bias_scores[0]}")
    print(f"Reward: {rewards[0]}")
