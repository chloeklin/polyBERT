"""Utility packages"""
import os
import ast
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

# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

class DictTensorDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }


"""Tokenise Masked Dataset"""

def load_tokenised_data(file_path):
    # Load tokenized data from file
    if os.path.isdir(file_path):  # Check if it's a directory since save_to_disk creates a directory
        dataset = load_from_disk(file_path)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return dataset
    return None


            
"""I/O"""
def load_txt_to_list(file_path):
    combined_list = []
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for line in file:
                # Convert the string representation of the list to an actual list
                line_list = ast.literal_eval(line.strip())
                combined_list.extend(line_list)  # Add items to the combined list
    else:
        print("File does not exist.")
    return combined_list
    
# def write_row_to_csv(file_path, row):
#     # Check if the file exists
#     file_exists = os.path.isfile(file_path)

#     # Open the file in append mode. If it doesn't exist, 'a' will create it
#     with open(file_path, mode='a', newline='') as file:
#         writer = csv.writer(file)

#         # If the file doesn't exist, write the header first (optional)
#         if not file_exists:
#             header = ['Pretrain size', 'f1-score']  # Adjust according to your needs
#             writer.writerow(header)

#         # Write the new row
#         writer.writerow(row)
def write_row_to_csv(file_path, row):
    # Initialize a flag to track if 'Pretrain size' is found
    updated = False
    file_exists = os.path.isfile(file_path)
    rows = []

    # If the file exists, read all rows into memory
    if file_exists:
        with open(file_path, mode="r", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)  # Read header, if it exists
            rows.append(header)  # Keep header for re-writing
            for existing_row in reader:
                # Check if 'Pretrain size' matches, update the row if it does
                if existing_row[0] == row[0]:
                    rows.append(row)  # Replace the row with the new one
                    updated = True
                else:
                    rows.append(existing_row)

    # If 'Pretrain size' not found, add the new row
    if not updated:
        if not file_exists:
            rows.append(['Pretrain size', 'f1-score'])  # Add header if file was missing
        rows.append(row)  # Add new row

    # Write all rows back to the file
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


    

def evaluate(size, batch_size):
    """Tokeniser"""
    tokeniser = DebertaV2Tokenizer(f"spm/spm_{size}.model",f"spm/spm_{size}.vocab")
    logging.info('Init tokeniser')
    
    """Model"""
    state_dict = torch.load(f"model_{size}_state_dict.pth")
    
    # Configuration for the DeBERTa model
    config = DebertaV2Config(
        vocab_size=265,
        hidden_size=600,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=512,
        pad_token_id=3
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DebertaV2ForMaskedLM(config=config)
    model.resize_token_embeddings(len(tokeniser))
    model.load_state_dict(state_dict)
    model.to(device)
    logging.info('Init model')

    # Use only padding without adding masks
    data_collator = DataCollatorWithPadding(tokenizer=tokeniser)
    
  
    
    """Mask dataset"""
    ground_truth = load_txt_to_list(f"truth_eval_{size}.txt")
    dataset = load_tokenised_data(f"data/tokenised_eval_{size}")
    logging.info('Loaded tokenised dataset')
    
    # Create a DataLoader to batch inputs
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=8)
    logging.info('Init dataloader')
    
    all_predicted_token_ids = []
    all_true_token_ids = tokeniser.convert_tokens_to_ids(ground_truth)
 
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get model outputs in one forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Vectorized mask position extraction
            mask_indices = (input_ids == tokeniser.mask_token_id)
            masked_logits = logits[mask_indices]
            predicted_token_ids = torch.argmax(masked_logits, dim=1).tolist()
            all_predicted_token_ids.extend(predicted_token_ids)
        

    
    """Calculate F1 Score"""
    # Convert ground truth tokens to IDs if not already done
    # Assuming ground_truth_ids are token IDs
    assert len(all_true_token_ids) == len(all_predicted_token_ids), \
        f"Mismatch in number of ground truth and predictions: {len(all_true_token_ids)} vs {len(all_predicted_token_ids)}"

    f1 = f1_score(all_true_token_ids, all_predicted_token_ids, average='micro')
    print(f"F1 Score: {f1}")

    """Write Results to CSV"""
    write_row_to_csv(f"evaluation_result.csv", [size, f1])




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
    # ngpus=args.ngpus
    batch=args.batch
    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)
    logging.basicConfig(level=logging.INFO)
    

    # Load tokenizer and model
    evaluate(size, batch)


if __name__ == '__main__':
    main()