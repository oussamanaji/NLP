import unittest
import torch
from src.ciat import CIAT

class TestCIAT(unittest.TestCase):
    def setUp(self):
        self.model_name = "CohereForAI/aya-23-8B"
        self.ciat = CIAT(self.model_name)

    def test_forward(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)
        
        causal_states, do_calculus = self.ciat(input_ids, attention_mask)
        
        self.assertEqual(causal_states.shape, (1, 10, self.ciat.base_model.config.n_embd))
        self.assertEqual(do_calculus.shape, (1, 10, 3))

if __name__ == '__main__':
    unittest.main()
