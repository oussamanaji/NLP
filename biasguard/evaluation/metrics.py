import torch
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter

def compute_perplexity(model, dataset, max_samples=1000):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            inputs = model.tokenizer(sample['response'], return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def compute_bleu(model, dataset, max_samples=1000):
    model.eval()
    bleu_scores = []
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            reference = sample['response'].split()
            generated = model.generate(sample['response'][:50], max_length=100)  # Use first 50 chars as prompt
            candidate = generated.split()
            
            bleu_score = sentence_bleu([reference], candidate)
            bleu_scores.append(bleu_score)
    
    return sum(bleu_scores) / len(bleu_scores)

def compute_diversity(model, dataset, max_samples=1000):
    model.eval()
    all_unigrams = Counter()
    all_bigrams = Counter()
    total_unigrams = 0
    total_bigrams = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            generated = model.generate(sample['response'][:50], max_length=100)
            tokens = generated.split()
            
            unigrams = tokens
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            
            all_unigrams.update(unigrams)
            all_bigrams.update(bigrams)
            total_unigrams += len(unigrams)
            total_bigrams += len(bigrams)
    
    distinct_1 = len(all_unigrams) / total_unigrams
    distinct_2 = len(all_bigrams) / total_bigrams
    
    return distinct_1, distinct_2

# Usage
if __name__ == "__main__":
    from models.actor_model import ActorModel
    from data.data_processing import load_and_process_datasets

    model = ActorModel()
    dataset = load_and_process_datasets()

    perplexity = compute_perplexity(model, dataset)
    bleu_score = compute_bleu(model, dataset)
    distinct_1, distinct_2 = compute_diversity(model, dataset)

    print(f"Perplexity: {perplexity}")
    print(f"BLEU Score: {bleu_score}")
    print(f"Distinct-1: {distinct_1}")
    print(f"Distinct-2: {distinct_2}")
