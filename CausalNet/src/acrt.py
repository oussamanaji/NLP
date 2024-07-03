import torch
import torch.nn as nn
import random

class ACRT(nn.Module):
    def __init__(self, base_model_n_embd):
        super(ACRT, self).__init__()
        self.adversarial_net = nn.Sequential(
            nn.Linear(base_model_n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def generate_adversarial_question(self, graph, original_question):
        nodes = list(graph.nodes())
        adversarial_type = random.choice(['swap_nodes', 'reverse_causality', 'add_spurious'])
        if adversarial_type == 'swap_nodes':
            a, b = random.sample(nodes, 2)
            return original_question.replace(a, b).replace(b, a)
        elif adversarial_type == 'reverse_causality':
            if "effect of" in original_question:
                parts = original_question.split("effect of")
                return f"{parts[0]}effect of{parts[1].split('on')[1]} on{parts[1].split('on')[0]}"
        # Implement more adversarial types
        return original_question

    def discriminate(self, representation):
        return self.adversarial_net(representation)
