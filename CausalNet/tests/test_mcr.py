import unittest
import torch
import networkx as nx
from src.mcr import MCR

class TestMCR(unittest.TestCase):
    def setUp(self):
        self.base_model_n_embd = 768
        self.mcr = MCR(self.base_model_n_embd)

    def test_encode_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        graph_hidden = self.mcr.encode_graph(G)
        self.assertEqual(graph_hidden.shape, (256,))

    def test_encode_text(self):
        text_embedding = torch.randn(1, 10, self.base_model_n_embd)
        text_hidden = self.mcr.encode_text(text_embedding)
        self.assertEqual(text_hidden.shape, (256,))

    def test_fuse_representations(self):
        graph_hidden = torch.randn(256)
        text_hidden = torch.randn(256)
        fused = self.mcr.fuse_representations(graph_hidden, text_hidden)
        self.assertEqual(fused.shape, (self.base_model_n_embd,))

if __name__ == '__main__':
    unittest.main()
