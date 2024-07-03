import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class CIAT(nn.Module):
    def __init__(self, base_model_name):
        super(CIAT, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.causal_layer = nn.Linear(self.base_model.config.n_embd, self.base_model.config.n_embd)
        self.do_calculus = nn.Linear(self.base_model.config.n_embd, 3)  # do, do not, conditional

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        causal_states = self.causal_layer(hidden_states)
        do_calculus = self.do_calculus(causal_states)
        return causal_states, do_calculus
