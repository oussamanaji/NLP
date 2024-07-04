import torch
from .base_model import BiasGuardBaseModel

class RewardModel(BiasGuardBaseModel):
    def __init__(self, model_id="CohereForAI/aya-23-8B"):
        super().__init__(model_id)
        self.reward_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        hidden_states = super().forward(input_ids, attention_mask)
        reward = self.reward_head(hidden_states[:, -1, :])
        return reward

    def compute_reward(self, text, bias_score):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            model_reward = self.forward(input_ids, attention_mask)
        
        # Combine model reward with bias score
        final_reward = model_reward - 10 * bias_score  # Penalize high bias
        return final_reward.item()

# Usage
if __name__ == "__main__":
    reward_model = RewardModel()
    reward = reward_model.compute_reward("Everyone should have equal opportunities regardless of their background.", 0.1)
    print(f"Computed reward: {reward}")
