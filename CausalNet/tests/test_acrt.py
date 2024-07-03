import unittest
import torch
import networkx as nx
from src.acrt import ACRT

class TestACRT(unittest.TestCase):
    def setUp(self):
        self.base_model_n_embd = 768
        self.acrt = ACRT(self.base_model_n_embd)

    def test_generate_adversarial_question(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        original_question = "Is there a direct effect of A on C?"
        adv_question = self.acrt.generate_adversarial_question(G, original_question)
        self.assertIsInstance(adv_question, str)
        self.assertNotEqual(adv_question, original_question)

    def test_discriminate(self):
        representation = torch.randn(1, self.base_model_n_embd)
        score = self.acrt.discriminate(representation)
        self.assertEqual(score.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
