import json
import random
from tqdm import tqdm

class CLEARBenchmark:
    def __init__(self, data_path):
        self.tasks = self.load_tasks(data_path)

    def load_tasks(self, data_path):
        with open(data_path, 'r') as f:
            return json.load(f)

    def evaluate_model(self, model, tokenizer, device):
        results = []
        for task in tqdm(self.tasks, desc="Evaluating CLEAR benchmark"):
            graph = task['graph']
            question = task['question']
            task_type = task['task_type']
            question_type = task['question_type']
            
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_token = torch.argmax(logits[0, -1, :]).item()
                predicted_answer = tokenizer.decode([predicted_token])
            
            results.append({
                'task_id': task['id'],
                'task_type': task_type,
                'question_type': question_type,
                'model_response': predicted_answer,
                'format_prompt': self.generate_format_prompt(task_type, question_type)
            })
        
        return results

    def generate_format_prompt(self, task_type, question_type):
        return f"Extract the answer for the {task_type} task with {question_type} question type."

    def split_results(self, results):
        split_results = {
            'FA': [], 'FO': [], 'HM': [], 'CS': [], 'YN': [], 'EX': []
        }
        for result in results:
            q_type = result['question_type']
            if q_type in split_results:
                split_results[q_type].append(result)
        return split_results
