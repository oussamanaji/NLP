import unittest
import torch
import networkx as nx
from src.ecrt import ECRT

class TestECRT(unittest.TestCase):
    def setUp(self):
        self.base_model_n_embd = 768
        self.vocab_size = 50000
        self.ecrt = ECRT(self.base_model_n_embd, self.vocab_size)

    def test_generate_explanation(self):
        causal_states = torch.randn(1, 10, self.base_model_n_embd)
        explanation_logits = self.ecrt.generate_explanation(causal_states)
        self.assertEqual(explanation_logits.shape, (1, self.vocab_size))

    def test_check_consistency(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        explanation = "A causes B, which in turn causes C."
        is_consistent = self.ecrt.check_consistency(explanation, G)
        self.assertIsInstance(is_consistent, bool)

if __name__ == '__main__':
    unittest.main()
