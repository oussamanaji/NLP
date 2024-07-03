import unittest
from src.hccl import HCCL

class TestHCCL(unittest.TestCase):
    def setUp(self):
        self.hccl = HCCL(num_levels=5)

    def test_generate_task(self):
        graph, question = self.hccl.generate_task()
        self.assertIsNotNone(graph)
        self.assertIsInstance(question, str)

    def test_update_level(self):
        initial_level = self.hccl.current_level
        self.hccl.update_level(0.9)  # High performance
        self.assertGreater(self.hccl.current_level, initial_level)

        self.hccl.current_level = 2
        self.hccl.update_level(0.4)  # Low performance
        self.assertEqual(self.hccl.current_level, 1)

if __name__ == '__main__':
    unittest.main()
