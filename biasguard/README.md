# BiasGuard: Bias Mitigation in NLP using Multi-Agent Deep Reinforcement Learning

## Overview
BiasGuard is a cutting-edge project designed to mitigate biases in AI-generated text. Leveraging the power of Cohere’s `CohereForAI/aya-23-8B` model, this project incorporates advanced techniques such as reinforcement learning, quantization, and Low-Rank Adaptation (LoRA) to enhance model performance and fairness.

## Key Features
- **Data Preparation**: Utilized Social Bias Frames, CrowS-Pairs, and a synthetic dataset generated using Cohere R+.
- **Model Architecture**: Fine-tuned with additional layers and dropout to prevent overfitting.
- **Reinforcement Learning**: Implemented Proximal Policy Optimization (PPO) for training.
- **Multi-Role Debates**: Evaluated responses across different personas to identify and mitigate biases.
- **Quantization**: Applied 4-bit quantization for efficient processing.

## Results
- **Perplexity**: Improved by 30% (35.2 to 24.8)
- **BLEU Score**: Increased by 38% (19.4 to 26.7)
- **Diversity Metrics**: 
  - **Distinct-1**: Increased from 0.33 to 0.49
  - **Distinct-2**: Increased from 0.28 to 0.41
- **Bias Scores**: Achieved a 42% reduction in detected bias levels

## Installation
Colab placholder for later

## Usage

- Prepare datasets and run the data preparation scripts.
- Configure the model architecture and training parameters.
- Execute the training script to fine-tune the model.
- Evaluate the model using the provided evaluation metrics scripts.

## Contributing
I welcome contributions to improve BiasGuard. Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to Cohere for their CohereForAI/aya-23-8B m
