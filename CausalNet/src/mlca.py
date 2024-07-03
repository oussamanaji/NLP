import torch
import torch.nn as nn

class MLCA(nn.Module):
    def __init__(self, base_model_n_embd):
        super(MLCA, self).__init__()
        self.abstraction_net = nn.Sequential(
            nn.Linear(base_model_n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.meta_learner = nn.GRU(input_size=128, hidden_size=256, batch_first=True)

    def abstract_causal_principle(self, causal_states):
        return self.abstraction_net(causal_states)

    def meta_learn(self, abstractions):
        _, hidden = self.meta_learner(abstractions)
        return hidden
