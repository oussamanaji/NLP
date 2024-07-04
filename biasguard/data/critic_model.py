import torch
from .base_model import BiasGuardBaseModel

class CriticModel(BiasGuardBaseModel):
    def __init__(self, model_id="CohereForAI/aya-23-8B"):
        super().__init__(model_id)
        self.bias_head = torch.nn.Linear(self.model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        hidden_states = super().forward(input_ids, attention_mask)
        bias_score = self.bias_head(hidden_states[:, -1, :])
        return torch.sigmoid(bias_score)

    def evaluate_bias(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            bias_score = self.forward(input_ids, attention_mask)
        return bias_score.item()

# Usage
if __name__ == "__main__":
    critic = CriticModel()
    bias_score = critic.evaluate_bias("Men are naturally better at leadership roles than women.")
    print(f"Bias score: {bias_score}")
