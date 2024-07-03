import torch
import torch.nn as nn
import random

class ECRT(nn.Module):
    def __init__(self, base_model_n_embd, vocab_size):
        super(ECRT, self).__init__()
        self.explanation_generator = nn.GRU(input_size=base_model_n_embd, hidden_size=512, batch_first=True)
        self.explanation_decoder = nn.Linear(512, vocab_size)

    def generate_explanation(self, causal_states):
        _, hidden = self.explanation_generator(causal_states)
        explanation_logits = self.explanation_decoder(hidden.squeeze(0))
        return explanation_logits

    def check_consistency(self, explanation, causal_graph):
        # Implement logic to check if the explanation is consistent with the causal graph
        # This is a placeholder implementation
        return random.random() > 0.2  # 80% chance of being consistent
