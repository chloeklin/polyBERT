"""Utility packages"""
import os
import csv

import argparse
"""Model setup"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
"""Evaluation"""
from sklearn.metrics import f1_score
"""Deepspeed"""
import lightning as L
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from polyBERT import polyBERT
from utils.data_utils import create_masked_test_set
from utils.model_utils import load_tokenizer, load_model, zero_checkpoint_to_fp32




def write_row_to_csv(file_path, row, header=None):
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode. If it doesn't exist, 'a' will create it
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first (optional)
        if not file_exists:
            writer.writerow(header)

        # Write the new row
        writer.writerow(row)


    

def evaluate(root_dir, size, psmiles_strings, batch_size, ngpus, csv_file):
    # save_path = f"./model_{size}_ds/last.ckpt"
    # output_path = f"./model_{size}_ds/last.pt"
    model_path = f"{root_dir}/pretrain_models/model_state_dict/model_{size}_state_dict.pth"
    # convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    if not os.path.isfile(model_path):
        zero_checkpoint_to_fp32(f"{root_dir}/pretrain_models/", size)


    # Init tokeniser and model
    tokeniser = load_tokenizer(f"{root_dir}/tokeniser/spm/", size)
    model = load_model(tokeniser, 'lightning', ckpt=model_path)
    
    # Mask 15% of tokens of each string in test data
    masked_psmiles, ground_truth = create_masked_test_set(tokeniser,psmiles_strings)
    
    # Tokenize the sentences
    inputs = tokeniser(masked_psmiles, return_tensors='pt', padding=True)
    
    # Create a DataLoader to batch inputs
    dataset = Dataset.load_from_disk(f"data/tokenized_{size}/train")
    
    dataset.set_format(type='torch', columns=['input_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=11)

    
    all_predicted_token_ids = []
    all_true_token_ids = tokeniser.convert_tokens_to_ids(ground_truth)
    
    trainer = Trainer(deterministic=True,
                      accelerator='gpu',
                      devices=ngpus,
                      strategy="deepspeed_stage_3",
                      enable_progress_bar=True,
                      precision=16)
    predictions = trainer.predict(model, dataloader)

    # Concatenate predictions
    for batch_predictions in predictions:
        all_predicted_token_ids.append(batch_predictions)

    # Compute F1 score (using token IDs for comparison)
    f1 = f1_score(all_true_token_ids, all_predicted_token_ids, average='micro')
    print(f"F1 score: {f1}")

    write_row_to_csv(csv_file, [size,f1], ['Pretrain size', 'f1-score'])




def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')

    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    ngpus=args.ngpus
    batch=args.batch
    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)

    # Load test dataset
    file_path = 'data/generated_polymer_smiles_dev.txt'
    csv_file = "masking_evaluation.csv"


    with open(file_path, 'r') as file:
        psmiles_strings = [line.strip() for line in file]

    # Load tokenizer and model
    evaluate(size, psmiles_strings, batch, ngpus, csv_file)


    # tokeniser = DebertaV2Tokenizer.from_pretrained('original_tok')
    # model = DebertaV2ForMaskedLM.from_pretrained('original_model')
    # f1_original = evaluate(model, tokeniser, psmiles_strings)
    # write_row_to_csv(csv_file, ['original(90M)',f1_original])

if __name__ == '__main__':
    main()