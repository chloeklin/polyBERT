import os
import random
from datasets import load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split

def _load_and_format(path, columns):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} does not exist.")
    dataset = load_from_disk(path)
    dataset.set_format(type='torch', columns=columns)
    return dataset

def load_masked_data(file_path):
    return _load_and_format(file_path, columns=['input_ids', 'attention_mask'])

def load_pretrain_data(base_path):
    train = _load_and_format(os.path.join(base_path, "train"), columns=['input_ids'])
    test = _load_and_format(os.path.join(base_path, "test"), columns=['input_ids'])
    return train, test

def create_masked_test_set(tokenizer, sentences, mask_prob=0.15):
    masked_sentences = []
    ground_truth = []

    for sentence in sentences:
        tokenized_input = tokenizer.tokenize(sentence)
        masked_sentence = tokenized_input.copy()  # Copy of tokenized sentence
        ground_truth_sentence = []

        for i, token in enumerate(tokenized_input):
            if random.random() < mask_prob:  # Mask with a certain probability (e.g., 15%)
                ground_truth.append(token)  # Store original token
                masked_sentence[i] = tokenizer.mask_token  # Replace with [MASK]

        masked_sentences.append(tokenizer.convert_tokens_to_string(masked_sentence))

    return masked_sentences, ground_truth


def train_val_test_split(root_dir: str, df: pd.DataFrame, filename: str, val_size=0.1, test_size=0.1):
    train_data, temp_data = train_test_split(df, test_size=(val_size+test_size), random_state=1)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=1)

    # Save the splits to new CSV files
    train_data.to_csv(f'{root_dir}/data/evaluation/train_{filename}.csv', index=False)
    val_data.to_csv(f'{root_dir}/data/evaluation/val_{filename}.csv', index=False)
    test_data.to_csv(f'{root_dir}/data/evaluation/test_{filename}.csv', index=False)
