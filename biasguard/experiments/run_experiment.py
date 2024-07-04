import torch
from tqdm import tqdm
from data.data_processing import load_and_process_datasets
from models.actor_model import ActorModel
from models.critic_model import CriticModel
from models.reward_model import RewardModel
from training.ppo_trainer import BiasGuardPPOTrainer
from training.multi_role_debates import MultiRoleDebateGenerator
from training.self_reflection import SelfReflectionModule
from evaluation.metrics import compute_perplexity, compute_bleu, compute_diversity
from evaluation.bias_evaluation import evaluate_overall_bias, evaluate_bias_categories
from utils.config import BiasGuardConfig
from utils.logging_utils import setup_logging, init_wandb, log_metrics
from utils.visualization import plot_training_progress, plot_bias_categories

def run_experiment(config):
    # Setup
    setup_logging()
    init_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    dataset = load_and_process_datasets()
    
    # Initialize models
    actor_model = ActorModel(config.model_id).to(device)
    critic_model = CriticModel(config.model_id).to(device)
    reward_model = RewardModel(config.model_id).to(device)
    
    # Initialize trainer and other modules
    trainer = BiasGuardPPOTrainer(actor_model, critic_model, reward_model, actor_model.tokenizer)
    debate_generator = MultiRoleDebateGenerator()
    self_reflection = SelfReflectionModule(actor_model)
    
    # Training loop
    metrics = {'step': [], 'loss': [], 'perplexity': [], 'bleu_score': [], 'bias_score': []}
    for epoch in range(config.num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        debate_prompts = debate_generator.generate_debate_topics(n=config.prompts_per_epoch)
        
        for step, prompt in enumerate(tqdm(debate_prompts)):
            # PPO training step
            stats, responses, bias_scores, rewards = trainer.train_step([prompt])
            
            # Self-reflection and improvement
            reflection = self_reflection.reflect_on_response(prompt, responses[0])
            improved_response = self_reflection.generate_improved_response(prompt, responses[0], reflection)
            
            # Log metrics
            metrics['step'].append(epoch * config.prompts_per_epoch + step)
            metrics['loss'].append(stats['loss'])
            metrics['perplexity'].append(compute_perplexity(actor_model, [{'response': improved_response}]))
            metrics['bleu_score'].append(compute_bleu(actor_model, [{'response': prompt}], [improved_response]))
            metrics['bias_score'].append(bias_scores[0])
            
            log_metrics(metrics, step=metrics['step'][-1])
            
        # Evaluate and save model periodically
        if (epoch + 1) % 5 == 0:
            evaluate_and_save(actor_model, dataset, config, epoch)
    
    # Final evaluation and save
    final_metrics = evaluate_and_save(actor_model, dataset, config, epoch, final=True)
    plot_training_progress(metrics)
    
    return final_metrics

def evaluate_and_save(model, dataset, config, epoch, final=False):
    perplexity = compute_perplexity(model, dataset)
    bleu_score = compute_bleu(model, dataset)
    distinct_1, distinct_2 = compute_diversity(model, dataset)
    overall_bias = evaluate_overall_bias(model, dataset)
    
    categories = ["gender", "race", "age", "religion", "nationality"]
    category_bias_scores = evaluate_bias_categories(model, dataset, categories)
    
    metrics = {
        "perplexity": perplexity,
        "bleu_score": bleu_score,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "overall_bias": overall_bias,
        **category_bias_scores
    }
    
    log_metrics(metrics, step=(epoch + 1) * config.prompts_per_epoch)
    plot_bias_categories(category_bias_scores)
    
    # Save model checkpoint
    if final:
        save_path = f"{config.output_dir}/final_model"
    else:
        save_path = f"{config.output_dir}/checkpoint_epoch{epoch+1}"
    model.save_pretrained(save_path)
    
    return metrics

if __name__ == "__main__":
    config = BiasGuardConfig()
    final_metrics = run_experiment(config)
    print("Experiment completed. Final metrics:", final_metrics)
