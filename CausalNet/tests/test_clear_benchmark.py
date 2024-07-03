import unittest
import json
import tempfile
from src.clear_benchmark import CLEARBenchmark

class TestCLEARBenchmark(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {
                "id": "task_001",
                "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]},
                "question": "Is there a direct effect of A on C?",
                "task_type": "direct_effect",
                "question_type": "YN"
            }
        ]
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        json.dump(self.sample_data, self.temp_file)
        self.temp_file.close()
        self.benchmark = CLEARBenchmark(self.temp_file.name)

    def test_load_tasks(self):
        self.assertEqual(len(self.benchmark.tasks), 1)
        self.assertEqual(self.benchmark.tasks[0]['id'], 'task_001')

    def test_generate_format_prompt(self):
        prompt = self.benchmark.generate_format_prompt("direct_effect", "YN")
        self.assertIn("direct_effect", prompt)
        self.assertIn("YN", prompt)

    def test_split_results(self):
        results = [
            {'question_type': 'YN', 'result': 'Yes'},
            {'question_type': 'FA', 'result': 'Path A-B-C'},
        ]
        split = self.benchmark.split_results(results)
        self.assertIn('YN', split)
        self.assertIn('FA', split)
        self.assertEqual(len(split['YN']), 1)
        self.assertEqual(len(split['FA']), 1)

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

if __name__ == '__main__':
    unittest.main()
