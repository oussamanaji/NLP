import unittest
import torch
from src.ckis import CKIS

class TestCKIS(unittest.TestCase):
    def setUp(self):
        self.ckis = CKIS()

    def test_integrate_and_retrieve_knowledge(self):
        causal_principle = torch.randn(128)
        confidence = 0.8
        self.ckis.integrate_knowledge(causal_principle, confidence)

        query = torch.randn(128)
        retrieved = self.ckis.retrieve_knowledge(query)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, (128,))

    def test_retrieve_nonexistent_knowledge(self):
        query = torch.randn(128)
        retrieved = self.ckis.retrieve_knowledge(query)
        self.assertIsNone(retrieved)

if __name__ == '__main__':
    unittest.main()
