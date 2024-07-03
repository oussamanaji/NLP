# CausalNet: Enterprise-Focused Causal AI Framework

## Overview

CausalNet is an innovative framework designed to enhance the causal reasoning capabilities of large language models, with a focus on enterprise applications. Built upon Cohere's aya-23-8B model, CausalNet implements a multi-faceted approach to improve AI's understanding and application of causal relationships in business contexts.

## Key Components

- HCCL: Hierarchical Causal Curriculum Learning
- CIAT: Causal Inference Augmented Transformer
- MCR: Multi-Modal Causal Representation
- ACRT: Adversarial Causal Robustness Training
- ECRT: Explainable Causal Reasoning Traces
- MLCA: Meta-Learning for Causal Abstraction
- CKIS: Causal Knowledge Integration System

## Installation

git clone https://github.com/oussamanaji/NLP/edit/main/CausalNet.git

cd CausalNet

pip install -r requirements.txt

## Usage

### Basic Usage

To run the CausalNet framework:

python main.py

### Advanced Usage

For detailed usage of individual components, refer to the Jupyter notebook:

notebooks/CausalNet_implementation.ipynb

## Project Structure

CausalNet/

├── notebooks/

│   └── CausalNet_implementation.ipynb

├── src/

│   ├── hccl.py

│   ├── ciat.py

│   ├── mcr.py

│   ├── acrt.py

│   ├── ecrt.py

│   ├── mlca.py

│   ├── ckis.py

│   └── clear_benchmark.py

├── data/

│   └── sample_causal_graphs.json

├── tests/

│   ├── test_hccl.py

│   ├── test_ciat.py

│   └── ...

├── results/

│   └── performance_reports/

├── config/

│   ├── model_config.yaml

│   └── training_config.yaml

├── scripts/

│   ├── evaluate.py

│   └── visualize_results.py

├── docs/

│   ├── api.md

│   └── usage_guide.md

├── requirements.txt

├── setup.py

├── main.py

├── README.md

└── LICENSE

## Features

- Enterprise-Focused Causal Learning: Tailored for business decision-making and strategy.
- Causal Inference Integration: Enhances the base model with specialized causal inference capabilities for enterprise scenarios.
- Multi-Modal Representation: Combines textual and graph-based causal information relevant to business contexts.
- Adversarial Training: Improves model robustness against real-world causal reasoning challenges.
- Explainable AI: Generates human-readable explanations for causal inferences in business terms.
- Meta-Learning: Abstracts and transfers causal knowledge across different business domains.
- Knowledge Integration: Dynamically updates and applies causal knowledge from various enterprise sources.

## Evaluation

CausalNet is evaluated using the CLEAR (Causal Language Evaluation And Reasoning) benchmark, adapted for enterprise scenarios. It assesses various aspects of causal understanding across different complexity levels in business contexts.

To run the evaluation:

python scripts/evaluate.py

## Results

Detailed performance reports, including business impact assessments, can be found in the `results/performance_reports/` directory.

## Contributing

I welcome contributions to the CausalNet project, especially those that enhance its applicability to enterprise scenarios. Please check the Contact section below or just submit a pull request!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project uses the Cohere aya-23-8B model as its foundation.
- Special thanks to the Cohere team for making their model available for research and enterprise applications.

## Contact

For any queries or discussions about enterprise applications of CausalNet, please open an issue in this repository or contact Mohamed Oussama Naji at mohamedoussama.naji@georgebrown.ca 
