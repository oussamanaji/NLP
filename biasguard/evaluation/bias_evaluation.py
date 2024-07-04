from tqdm import tqdm

def evaluate_overall_bias(model, dataset, max_samples=1000):
    model.eval()
    total_bias = 0
    num_samples = min(len(dataset), max_samples)
    
    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Evaluating bias")):
        if i >= max_samples:
            break
        
        generated = model.generate(sample['response'][:50], max_length=100)
        bias_score = model.critic_model.evaluate_bias(generated)
        total_bias += bias_score
    
    average_bias = total_bias / num_samples
    return average_bias

def evaluate_bias_categories(model, dataset, categories, max_samples=1000):
    model.eval()
    bias_scores = {category: [] for category in categories}
    num_samples = min(len(dataset), max_samples)
    
    for i, sample in enumerate(tqdm(dataset, total=num_samples, desc="Evaluating bias categories")):
        if i >= max_samples:
            break
        
        for category in categories:
            prompt = f"Considering {category}, {sample['response'][:50]}"
            generated = model.generate(prompt, max_length=100)
            bias_score = model.critic_model.evaluate_bias(generated)
            bias_scores[category].append(bias_score)
    
    average_bias_scores = {category: sum(scores) / len(scores) for category, scores in bias_scores.items()}
    return average_bias_scores

# Usage
if __name__ == "__main__":
    from models.actor_model import ActorModel
    from data.data_processing import load_and_process_datasets

    model = ActorModel()
    dataset = load_and_process_datasets()

    overall_bias = evaluate_overall_bias(model, dataset)
    print(f"Overall Bias Score: {overall_bias}")

    categories = ["gender", "race", "age", "religion", "nationality"]
    category_bias_scores = evaluate_bias_categories(model, dataset, categories)
    for category, score in category_bias_scores.items():
        print(f"Bias Score for {category}: {score}")
