from sentence_transformers import SentenceTransformer, util
import torch
import json

class InformationRetrievalModel:
    def __init__(self, model_name='fine-tuned/NFCorpus-256-24-gpt-4o-2024-05-13-203779', evidence_file='evidence.json'):
        self.model = SentenceTransformer(model_name)
        self.evidence_embeddings = None
        self.evidence_texts = None
        self.load_evidence(evidence_file)

    def load_evidence(self, evidence_file):
        with open(evidence_file, 'r') as f:
            evidence_data = json.load(f)
        self.evidence_texts = evidence_data['evidence']
        self.evidence_embeddings = self.model.encode(self.evidence_texts, convert_to_tensor=True)

    def retrieve_evidence(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.evidence_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return [self.evidence_texts[i] for i in top_results.indices]
