import random
import networkx as nx

class HCCL:
    def __init__(self, num_levels=5):
        self.num_levels = num_levels
        self.current_level = 0

    def generate_task(self):
        complexity = self.current_level + 1
        graph = self.generate_causal_graph(complexity)
        question = self.generate_question(graph, complexity)
        return graph, question

    def generate_causal_graph(self, complexity):
        G = nx.DiGraph()
        num_nodes = complexity * 2
        for i in range(num_nodes):
            G.add_node(chr(65 + i))  # Add nodes A, B, C, ...
        for _ in range(complexity * 3):
            a, b = random.sample(list(G.nodes()), 2)
            G.add_edge(a, b)
        return G

    def generate_question(self, graph, complexity):
        question_types = ['direct_effect', 'indirect_effect', 'confounding', 'collider', 'mediator']
        question_type = random.choice(question_types[:complexity])
        nodes = list(graph.nodes())
        if question_type == 'direct_effect':
            a, b = random.sample(nodes, 2)
            return f"Is there a direct causal effect of {a} on {b}?"
        elif question_type == 'indirect_effect':
            a, b, c = random.sample(nodes, 3)
            return f"Is there an indirect causal effect of {a} on {c} through {b}?"
        # Add more question types as complexity increases
        return "Is there a causal relationship between A and B?"

    def update_level(self, performance):
        if performance > 0.8:
            self.current_level = min(self.current_level + 1, self.num_levels - 1)
        elif performance < 0.5:
            self.current_level = max(self.current_level - 1, 0)
