import torch

class CKIS:
    def __init__(self):
        self.knowledge_base = {}

    def integrate_knowledge(self, causal_principle, confidence):
        key = hash(str(causal_principle.detach().cpu().numpy()))
        if key not in self.knowledge_base or confidence > self.knowledge_base[key][1]:
            self.knowledge_base[key] = (causal_principle, confidence)

    def retrieve_knowledge(self, query):
        similarities = [(k, torch.cosine_similarity(query, v[0], dim=0)) for k, v in self.knowledge_base.items()]
        if similarities:
            best_match = max(similarities, key=lambda x: x[1])
            return self.knowledge_base[best_match[0]][0]
        return None
