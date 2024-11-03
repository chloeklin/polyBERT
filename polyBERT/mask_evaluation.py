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
from datasets import load_dataset, Dataset, load_from_disk
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DebertaV2Config, DataCollatorForLanguageModeling, pipeline, DataCollatorWithPadding
from lightning.pytorch import seed_everything
from sklearn.metrics import f1_score
from transformers.pipelines.pt_utils import KeyDataset

"""Mask Dataset"""
def mask_sentences(sentences, tokenizer, mask_prob=0.15):
    masked = []
    truth = []
    for sentence in sentences:
        masked_string, ground_truth_sentence = mask_sentence(sentence, tokenizer, mask_prob)
        masked.append(masked_string)
        truth.append(ground_truth_sentence)
    return masked, truth
        
def mask_sentence(sentence, tokenizer, mask_prob):
    tokenized_input = tokenizer.tokenize(sentence)
    masked_sentence = tokenized_input.copy()
    ground_truth_sentence = []

    for i, token in enumerate(tokenized_input):
        if random.random() < mask_prob:
            # Save the original token as ground truth
            ground_truth_sentence.append(token)
            # Replace token with [MASK]
            masked_sentence[i] = tokenizer.mask_token

    # Ensure lengths match
    assert masked_sentence.count(tokenizer.mask_token) == len(ground_truth_sentence), \
        f"Mismatch: {masked_sentence.count(tokenizer.mask_token)} masked tokens, {len(ground_truth_sentence)} ground truth tokens"

    # Convert masked tokens back to a string
    masked_string = tokenizer.convert_tokens_to_string(masked_sentence)
    return masked_string, ground_truth_sentence

"""Tokenise Masked Dataset"""
def tokenise(data, tokeniser):
    # Tokenise the text and return input_ids and attention_mask
    res = tokeniser(data['text'], return_attention_mask=True, return_tensors="pt",padding=True)

    return {'input_ids': res['input_ids'], 'attention_mask': res['attention_mask']}

def tokenise_sentences(data_files, tokeniser):
    # Load the text dataset from a text file
    dataset = load_dataset('text', data_files=data_files, cache_dir='eval_cache/')
    dataset = dataset['train']  # Select the 'train' split explicitly
    # Tokenise the dataset by mapping tokenise() to each batch
    dataset = dataset.map(lambda x: tokenise(x, tokeniser), batched=True, batch_size=10_000, num_proc=10) 
    
    return dataset

def save_list_to_txt(file_path, data_list):
    with open(file_path, "w") as file:
        for item in data_list:
            file.write(f"{item}\n")  # Write each item on a new line
            
# def mask_sentence(sentence, tokenizer, mask_prob=0.15):
#     tokenized_input = tokenizer.tokenize(sentence)
#     masked_sentence = tokenized_input.copy()
#     ground_truth_sentence = []

#     for i, token in enumerate(tokenized_input):
#         if random.random() < mask_prob:
#             ground_truth_sentence.append(token)
#             masked_sentence[i] = tokenizer.mask_token

#     return tokenizer.convert_tokens_to_string(masked_sentence), ground_truth_sentence

# def mask_sentence_worker(args):
#     sentence, tokenizer, mask_prob = args
#     return mask_sentence(sentence, tokenizer, mask_prob)

# def create_masked_test_set_parallel(tokenizer, sentences, mask_prob=0.15, num_workers=32):
#     # Prepare arguments for each sentence
#     args = [(sentence, tokenizer, mask_prob) for sentence in sentences]

#     # Use multiprocessing to apply mask_sentence in parallel
#     with mp.Pool(num_workers) as pool:
#         results = pool.map(mask_sentence_worker, args)

#     masked_sentences, ground_truth = zip(*results)
#     return list(masked_sentences), list(ground_truth) #itertools.chain(*ground_truth)





def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    # parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')

    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    seed_everything(1, workers=True)
    file_path = 'data/generated_polymer_smiles_dev.txt'
    with open(file_path, 'r') as file:
        psmiles_strings = [line.strip() for line in file]

    """Tokeniser"""
    tokeniser = DebertaV2Tokenizer(f"spm/spm_{size}.model",f"spm/spm_{size}.vocab")
    logging.info('Init tokeniser')

    masked_psmiles, ground_truth = mask_sentences(psmiles_strings,tokeniser)
    save_list_to_txt(f"data/masked_eval_{size}.txt", masked_psmiles)
    save_list_to_txt(f"data/truth_eval_{size}.txt", ground_truth)
    # Tokenize the sentences
    dataset = tokenise_sentences(f"data/masked_eval_{size}.txt", tokeniser)
    dataset.save_to_disk(f'data/tokenised_eval_{size}')
    

if __name__ == '__main__':
    main()