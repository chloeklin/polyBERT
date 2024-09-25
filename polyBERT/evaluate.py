import os
import csv
import torch
import random
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer

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

def write_row_to_csv(file_path, row):
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode. If it doesn't exist, 'a' will create it
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first (optional)
        if not file_exists:
            header = ['Pretrain size', 'f1-score']  # Adjust according to your needs
            writer.writerow(header)

        # Write the new row
        writer.writerow(row)

def evaluate(tokenizer, model, psmiles_strings, device, batch_size=64):
    
    # Set the model to evaluation mode
    model.eval()
    
    # Mask 15% of tokens of each string in test data
    masked_psmiles, ground_truth = create_masked_test_set(tokenizer,psmiles_strings)
    
    # Tokenize the sentences
    inputs = tokenizer(masked_psmiles, return_tensors='pt', padding=True)
    
    # Create a DataLoader to batch inputs
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_predicted_token_ids = []
    all_true_token_ids = tokenizer.convert_tokens_to_ids(ground_truth)
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        # Run inference to get predictions for masked tokens
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits

        # Get the predicted token IDs for the masked positions
        masked_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        predicted_token_ids = predictions[masked_indices].argmax(dim=-1)
        
        # Collect all predicted token IDs
        all_predicted_token_ids.extend(predicted_token_ids.cpu().numpy())

    # Compute F1 score (using token IDs for comparison)
    f1 = f1_score(all_true_token_ids, all_predicted_token_ids, average='micro')
    
    return f1





# Load test dataset
file_path = 'data/generated_polymer_smiles_dev.txt'
csv_file = "masking_evaluation.csv"

with open(file_path, 'r') as file:
    psmiles_strings = [line.strip() for line in file]
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
size = '1M'
tokenizer = DebertaV2Tokenizer(f"spm_{size}.model",f"spm_{size}.vocab")
model = DebertaV2ForMaskedLM.from_pretrained(f'model_{size}_final/').to(device)
f1_1M = evaluate(tokenizer, model, psmiles_strings, device)
write_row_to_csv(csv_file, [size,f1_1M])

# size = '5M'
# tokenizer = DebertaV2Tokenizer(f"spm_{size}.model",f"spm_{size}.vocab")
# model = DebertaV2ForMaskedLM.from_pretrained(f'model_{size}_final/').to(device)
# f1_5M = evaluate(tokenizer, model, psmiles_strings, device)
# write_row_to_csv(csv_file, [size,f1_5M])

# tokenizer = DebertaV2Tokenizer.from_pretrained('original_tok')
# model = DebertaV2ForMaskedLM.from_pretrained('original_model').to(device)
# f1_original = evaluate(tokenizer, model, psmiles_strings, device)
# write_row_to_csv(csv_file, ['original(90M)',f1_original])
