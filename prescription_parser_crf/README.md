# CRF-based Prescription Parser

This project introduces an intelligent prescription parsing system that utilizes Conditional Random Fields (CRF) to accurately extract and classify relevant entities from medical prescriptions.

## Features
- Accurately identifies and extracts key elements such as medication names, dosages, frequencies, and durations from prescription texts
- Captures sequential dependencies and contextual information using Conditional Random Fields (CRF)
- Demonstrates expertise in named entity recognition, feature engineering, and model evaluation
- Achieves high precision and recall scores, outperforming traditional rule-based approaches
- Streamlines the process of extracting structured information from unstructured prescription data

## Technical Skills
- Natural Language Processing (NLP)
- Named Entity Recognition (NER)
- Conditional Random Fields (CRF)
- Feature Engineering
- Model Evaluation
- Python
- Libraries: sklearn-crfsuite, nltk, pandas

## Applications
- Electronic Health Record Systems
- Clinical Decision Support
- Medication Reconciliation
- Pharmacovigilance

## Methodology
1. **Data Preprocessing**: The prescription dataset is preprocessed, including tokenization, lowercasing, and removing special characters.
2. **Feature Engineering**: Relevant features, such as word embeddings, part-of-speech tags, and contextual information, are extracted from the prescription text to capture the characteristics of each entity.
3. **CRF Model Training**: The CRF model is trained using the engineered features and labeled prescription data, learning the sequential dependencies and entity patterns.
4. **Model Evaluation**: The trained CRF model is evaluated using metrics such as precision, recall, and F1-score to assess its performance in extracting and classifying prescription entities.
5. **Prescription Parsing**: The trained CRF model is applied to new prescription texts to extract and classify the relevant entities accurately.

## Results
- The CRF-based Prescription Parser achieved high precision and recall scores in extracting and classifying prescription entities, outperforming traditional rule-based approaches.
- The model successfully captured the sequential dependencies and contextual information present in prescription texts, enabling accurate identification of medication names, dosages, frequencies, and durations.
- The feature engineering techniques employed significantly improved the model's performance, incorporating relevant information such as word embeddings and part-of-speech tags.
- The project demonstrated the effectiveness of CRF in handling the complex nature of prescription data and extracting structured information for downstream applications.

## Getting Started
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hVqoIPHZDKOpypZY9dYDo5ajZ3W9Ll6I?usp=sharing)

1. Prepare your prescription dataset.
2. Follow the instructions in the notebook to train the CRF-based prescription parser and evaluate its performance.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. This means you are free to use, modify, and distribute the software, as long as you include the original license and copyright notice in any copies or substantial portions of the software.

## Contact
For any inquiries or collaborations, please contact:
- Mohamed Oussama NAJI
- LinkedIn: [Mohamed Oussama Naji](https://www.linkedin.com/in/oussamanaji/)
