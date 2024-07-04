# BiasGuard: Bias Mitigation in NLP using Multi-Agent Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![GitHub issues](https://img.shields.io/github/issues/oussamanaji/NLP)](https://github.com/oussamanaji/NLP/issues)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ufviUZ5k7zL8kB6VF6OJ_AgZ5tgHLgc?usp=sharing)

## 📚 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgements](#-acknowledgements)

## 🌟 Overview

BiasGuard is a cutting-edge project designed to mitigate biases in AI-generated text. Leveraging the power of Cohere's `CohereForAI/aya-23-8B` model, this project incorporates advanced techniques such as reinforcement learning, quantization, and Low-Rank Adaptation (LoRA) to enhance model performance and fairness.

Check out my [Presentation](https://docs.google.com/presentation/d/1x1HOh2n9KndEBkGM42DVgofS9bxkUNEi-vQu4x-3OOg/edit?usp=sharing) for a detailed overview of the project!

In an era where AI systems increasingly influence decision-making processes across various domains, addressing and mitigating biases in these systems is crucial. BiasGuard aims to contribute to the development of more equitable and fair AI by providing a comprehensive framework for bias detection, evaluation, and mitigation in large language models.

## 🚀 Key Features

- **Data Preparation**: Utilized Social Bias Frames, CrowS-Pairs, and a synthetic dataset generated using Cohere R+.
- **Model Architecture**: Fine-tuned with additional layers and dropout to prevent overfitting.
- **Reinforcement Learning**: Implemented Proximal Policy Optimization (PPO) for training.
- **Multi-Role Debates**: Evaluated responses across different personas to identify and mitigate biases.
- **Quantization**: Applied 4-bit quantization for efficient processing.
- **Multi-faceted Bias Mitigation**: Combines deep reinforcement learning, quantization, and Low-Rank Adaptation (LoRA) for effective bias reduction.
- **Self-Reflection Mechanism**: Implements a novel approach for the model to critique and improve its own outputs.
- **Comprehensive Evaluation Metrics**: Includes perplexity, BLEU score, diversity metrics, and specialized bias evaluation.

## 📁 Project Structure

```
biasguard/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_processing.py
├── models/
│   ├── actor_model.py
│   ├── critic_model.py
│   ├── reward_model.py
│   └── base_model.py
├── training/
│   ├── ppo_trainer.py
│   ├── multi_role_debates.py
│   └── self_reflection.py
├── evaluation/
│   ├── metrics.py
│   └── bias_evaluation.py
├── utils/
│   ├── config.py
│   ├── logging_utils.py
│   └── visualization.py
├── experiments/
│   ├── experiment_configs/
│   └── run_experiment.py
├── notebooks/
│   └── BiasGuard.ipynb
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠 Installation

1. Clone the repository:
   ```
   git clone https://github.com/oussamanaji/NLP.git
   cd NLP/biasguard
   ```

2. Create a virtual environment:
   ```
   python -m venv biasguard-env
   source biasguard-env/bin/activate  # On Windows, use `biasguard-env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install the project in editable mode:
   ```
   pip install -e .
   ```

## 🖥 Usage

To run a BiasGuard experiment:

1. Configure your experiment parameters in `utils/config.py`.
2. Run the main experiment script:
   ```
   python experiments/run_experiment.py
   ```

For a detailed walkthrough of the BiasGuard system, refer to the Jupyter notebook at `notebooks/BiasGuard.ipynb`.

You can also explore the project using our Google Colab notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ufviUZ5k7zL8kB6VF6OJ_AgZ5tgHLgc?usp=sharing)

[Note: The Colab notebook is still in progress]

## 🧠 Methodology

BiasGuard employs a sophisticated approach to bias mitigation:

1. **Data Preparation**: Utilizes diverse datasets including Social Bias Frames, CrowS-Pairs, and a custom synthetic dataset generated using Cohere R+.
2. **Model Architecture**: Builds upon the CohereForAI/aya-23-8B model, enhanced with custom layers and LoRA fine-tuning.
3. **Training Process**: 
   - Implements Proximal Policy Optimization (PPO) for reinforcement learning.
   - Utilizes multi-role debates to generate diverse perspectives.
   - Incorporates a self-reflection mechanism for continuous improvement.
4. **Evaluation**: Employs a comprehensive set of metrics including perplexity, BLEU score, diversity measures, and specialized bias evaluation techniques.

## 📊 Results

Our experiments demonstrate significant improvements in bias mitigation and overall performance:

- **Bias Reduction**: 42% decrease in detected bias levels post-training.
- **Perplexity**: Improved by 30% (from 35.2 to 24.8).
- **BLEU Score**: Increased by 38% (from 19.4 to 26.7).
- **Diversity Metrics**:
  - **Distinct-1**: Increased from 0.33 to 0.49 (48% improvement)
  - **Distinct-2**: Increased from 0.28 to 0.41 (46% improvement)

For detailed results and visualizations, refer to the `experiments/results` directory.

## 👥 Contributing

We welcome contributions to improve BiasGuard! Please fork the repository and submit a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Mohamed Oussama Naji**
- Email: mohamedoussama.naji@georgebrown.ca
- GitHub: [@oussamanaji](https://github.com/oussamanaji)
- Project Link: [https://github.com/oussamanaji/NLP/tree/main/biasguard](https://github.com/oussamanaji/NLP/tree/main/biasguard)

## 🙏 Acknowledgements

Special thanks to:
- Cohere for their CohereForAI/aya-23-8B model, which formed the foundation of this project.
- [Hugging Face](https://huggingface.co/) for their transformers library and datasets.
- [Allen Institute for AI](https://allenai.org/) for the Social Bias Frames dataset.
- [New York University Machine Learning for Language Lab](https://wp.nyu.edu/ml2/) for the CrowS-Pairs dataset.

Your support and contributions to the field have been instrumental in making projects like BiasGuard possible.

---

BiasGuard is part of ongoing research in AI ethics and bias mitigation. We are committed to advancing the field of responsible AI and welcome collaboration and feedback from the community.
