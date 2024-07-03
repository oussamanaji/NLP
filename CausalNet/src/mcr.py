import torch
import torch.nn as nn
import networkx as nx

class MCR(nn.Module):
    def __init__(self, base_model_n_embd):
        super(MCR, self).__init__()
        self.graph_encoder = nn.GRU(input_size=64, hidden_size=256, batch_first=True)
        self.text_encoder = nn.GRU(input_size=base_model_n_embd, hidden_size=256, batch_first=True)
        self.fusion_layer = nn.Linear(512, base_model_n_embd)

    def encode_graph(self, graph):
        adj_matrix = nx.adjacency_matrix(graph).todense()
        graph_embedding = torch.FloatTensor(adj_matrix).unsqueeze(0)
        _, graph_hidden = self.graph_encoder(graph_embedding)
        return graph_hidden.squeeze(0)

    def encode_text(self, text_embedding):
        _, text_hidden = self.text_encoder(text_embedding)
        return text_hidden.squeeze(0)

    def fuse_representations(self, graph_hidden, text_hidden):
        combined = torch.cat((graph_hidden, text_hidden), dim=-1)
        return self.fusion_layer(combined)
