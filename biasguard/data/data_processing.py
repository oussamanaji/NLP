import pandas as pd
from datasets import load_dataset, Dataset
import pyarrow as pa

def load_and_process_datasets():
    # Load datasets
    sb_dataset = load_dataset("allenai/social_bias_frames")['train']
    crows_pairs = load_dataset("nyu-mll/crows_pairs")['test']
    synthetic_df = pd.read_csv('Expanded_Bias_Trap_Dataset.csv')

    # Process Social Bias Frames
    def map_sb_bias_label(example):
        label = example['offensiveYN']
        if label == 0.0:
            return {'bias_label': 0, 'response': example['post']}
        elif label == 0.5:
            return {'bias_label': 1, 'response': example['post']}
        elif label == 1.0:
            return {'bias_label': 2, 'response': example['post']}
        else:
            return None

    sb_dataset = sb_dataset.map(map_sb_bias_label)
    sb_dataset = sb_dataset.filter(lambda x: x is not None)

    # Process CrowS-Pairs
    def map_crows_bias_type(bias_type):
        bias_map = {
            'race-color': 0, 'gender': 1, 'age': 2,
            'religion': 3, 'socioeconomic': 4, 'sexual-orientation': 5,
            'nationality': 6, 'disability': 7, 'physical-appearance': 8
        }
        return bias_map.get(bias_type, 9)  # 9 for any unrecognized type

    crows_df = pd.DataFrame({
        'response': crows_pairs['sent_more'],
        'bias_label': [map_crows_bias_type(bt) for bt in crows_pairs['bias_type']]
    })

    # Process synthetic dataset
    synthetic_df = synthetic_df.rename(columns={'bias_score': 'bias_label'})

    # Combine datasets
    combined_df = pd.concat([
        synthetic_df[['response', 'bias_label']],
        sb_dataset.to_pandas()[['response', 'bias_label']],
        crows_df
    ])

    # Remove any potential NaN values and reset index
    combined_df = combined_df.dropna().reset_index(drop=True)

    # Convert to a Hugging Face Dataset
    return Dataset(pa.Table.from_pandas(combined_df))

def process_data(example, tokenizer):
    return tokenizer(example['response'], padding='max_length', truncation=True, max_length=512)

# Usage
if __name__ == "__main__":
    processed_dataset = load_and_process_datasets()
    print(f"Combined dataset size: {len(processed_dataset)}")
    print(processed_dataset[0]) 
