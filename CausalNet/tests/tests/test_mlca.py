import unittest
import torch
from src.mlca import MLCA

class TestMLCA(unittest.TestCase):
    def setUp(self):
        self.base_model_n_embd = 768
        self.mlca = MLCA(self.base_model_n_embd)

    def test_abstract_causal_principle(self):
        causal_states = torch.randn(1, self.base_model_n_embd)
        abstraction = self.mlca.abstract_causal_principle(causal_states)
        self.assertEqual(abstraction.shape, (1, 128))

    def test_meta_learn(self):
        abstractions = torch.randn(1, 5, 128)  # Batch of 5 abstractions
        meta_hidden = self.mlca.meta_learn(abstractions)
        self.assertEqual(meta_hidden.shape, (1, 256))

if __name__ == '__main__':
    unittest.main()
