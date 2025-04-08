"""Utility packages"""
import argparse
import logging
""""Torch and HuggingFace"""
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from lightning.pytorch import seed_everything
from sklearn.metrics import f1_score
from utils.io_utils import load_txt_to_list, write_row_to_csv
from utils.data_utils import load_masked_data
from utils.model_utils import load_tokenizer, load_model


def evaluate(root_dir, size, batch_size):
    """Tokeniser"""
    tokeniser = load_tokenizer(f"{root_dir}/tokeniser/spm/", size)
    logging.info('Init tokeniser')
    
    """Model"""
    model = load_model(tokeniser, 'torch', f"{root_dir}/pretrain_models/model_state_dict/model_{size}_state_dict.pth")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    logging.info('Init model')

    """Dataset"""
    ground_truth = load_txt_to_list(f"{root_dir}/data/evaluation/MLM/truth_eval_{size}.txt")
    dataset = load_masked_data(f"{root_dir}/data/evaluation/MLM/tokenised_eval_{size}")
    logging.info('Loaded tokenised dataset')
    
    data_collator = DataCollatorWithPadding(tokenizer=tokeniser)
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
    parser.add_argument('--root_dir', type=str, help='Root directory of repository')
    parser.add_argument('--size', type=str, help='Pretraining size')
    # parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)
    logging.basicConfig(level=logging.INFO)

    # Load tokenizer and model
    evaluate(args.root_dir, args.size, args.batch)


if __name__ == '__main__':
    main()