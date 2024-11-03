"""Utility packages"""
import os
import csv
import random
import argparse
import logging
import cProfile
import itertools
import multiprocessing as mp
""""Torch and HuggingFace"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DebertaV2Config, DataCollatorForLanguageModeling
from lightning.pytorch import seed_everything

    
def load_csv_to_list(file_path):
    # Check if the file exists
    if os.path.isfile(file_path):
        # If it exists, load it as a list
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data
    else:
        print("File does not exist.")
        return None


def tokenise_sentence(sentence):
    global tokeniser  # Ensure tokeniser is accessible in the worker processes
    return tokeniser(sentence, return_tensors='pt', padding=True)


def tokenise_sentences(data_files, tokeniser):
    batch_size = 128  # Define batch size
    dataset = dataset = load_dataset("csv", data_files="path/to/your_file.csv")

    dataset = dataset.map(tokenize, batched=True, batch_size=10_000, num_proc=10) 
    dataset.save_to_disk(f'data/tokenized_{size}')



def tokenise_sentences_parallel(sentences, tokeniser_instance, num_workers=8):
    global tokeniser  # Set the tokeniser globally for use in each worker
    tokeniser = tokeniser_instance
    
    # Use multiprocessing to tokenise sentences
    with mp.Pool(num_workers) as pool:
        tokenised_sentences = pool.map(tokenise_sentence, sentences)
        
    return tokenised_sentences

def save_tokenised_data(tokenised_data, file_path):
    # Flatten batched tensors and save them as individual tensors
    input_ids = torch.cat([item['input_ids'] for item in tokenised_data])
    attention_mask = torch.cat([item['attention_mask'] for item in tokenised_data])
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, file_path)
    data = torch.load(file_path)
    dataset = DictTensorDataset(data['input_ids'], data['attention_mask'])
    return dataset
    
def load_tokenised_data(file_path):
    # Load tokenized data from file
    dataset = None
    if os.path.isfile(file_path):
        data = torch.load(file_path)
        dataset = DictTensorDataset(data['input_ids'], data['attention_mask'])
    return dataset  

    
    
    
def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')


    # Parse the arguments
    args = parser.parse_args()
    size=args.size

    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)
    logging.basicConfig(level=logging.INFO)
    

    tokeniser = DebertaV2Tokenizer(f"spm/spm_{size}.model",f"spm/spm_{size}.vocab")
    logging.info('Init tokeniser')
    
    """Mask dataset"""
    # check if masked dataset exist
    masked_psmiles = load_csv_to_list(f"masked_{size}.csv")

    if masked_psmiles:
        masked_psmiles = list(itertools.chain.from_iterable(masked_psmiles))
        logging.info('Masked dataset loaded')
    else:
        # Mask 15% of tokens of each string in test data
        print("No data to display.")
        masked_psmiles, ground_truth = create_masked_test_set_parallel(tokeniser,psmiles_strings)
        logging.info('Masked dataset done')
        with open(f"masked_{size}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(masked_psmiles)
        with open(f"truth_{size}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(ground_truth)


    dataset = load_tokenised_data(f"tokenised_data_{size}.pt")
    if not dataset:
        # Tokenize the sentences
        tokenised_data = tokenise_sentences_parallel(masked_psmiles[:10], tokeniser, num_workers=32)
        # Save to disk
        dataset = save_tokenised_data(tokenised_data, f"tokenised_data_{size}.pt")


if __name__ == '__main__':
    main()