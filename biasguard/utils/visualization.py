import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_training_progress(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress", fontsize=16)

    # Plot loss
    axes[0, 0].plot(metrics['step'], metrics['loss'])
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")

    # Plot perplexity
    axes[0, 1].plot(metrics['step'], metrics['perplexity'])
    axes[0, 1].set_title("Perplexity")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Perplexity")

    # Plot BLEU score
    axes[1, 0].plot(metrics['step'], metrics['bleu_score'])
    axes[1, 0].set_title("BLEU Score")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("BLEU Score")

    # Plot bias score
    axes[1, 1].plot(metrics['step'], metrics['bias_score'])
    axes[1, 1].set_title("Bias Score")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Bias Score")

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()

def plot_bias_categories(bias_scores):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(bias_scores.keys()), y=list(bias_scores.values()))
    plt.title("Bias Scores by Category")
    plt.xlabel("Category")
    plt.ylabel("Bias Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bias_categories.png")
    plt.close()

# Usage
if __name__ == "__main__":
    # Example data
    metrics = {
        'step': list(range(100)),
        'loss': [1 - i*0.01 for i in range(100)],
        'perplexity': [10 - i*0.1 for i in range(100)],
        'bleu_score': [i*0.01 for i in range(100)],
        'bias_score': [0.5 - i*0.005 for i in range(100)]
    }
    plot_training_progress(metrics)

    bias_scores = {
        'gender': 0.3,
        'race': 0.25,
        'age': 0.2,
        'religion': 0.15,
        'nationality': 0.1
    }
    plot_bias_categories(bias_scores)
