import os
import csv
import torch
import random
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, DebertaV2ForMaskedLM, DebertaV2Tokenizer

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

def evaluate(tokenizer, model, psmiles_strings, device):
    
    # Set the model to evaluation mode
    model.eval()
    
    # Mask 15% of tokens of each string in test data
    masked_psmiles, ground_truth = create_masked_test_set(tokenizer,psmiles_strings)
    
    # Tokenize the sentences
    inputs = tokenizer(masked_psmiles, return_tensors='pt', padding=True)
    inputs = inputs.to(device)
    
    # Run inference to get predictions for masked tokens
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Get the predicted token IDs for the masked positions
    masked_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    predicted_token_ids = predictions[masked_indices].argmax(dim=-1)
    
    # Convert predicted token IDs back to words
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
    
    # Convert true tokens to token IDs
    true_token_ids = tokenizer.convert_tokens_to_ids(ground_truth)
    
    # Compute F1 score (using token IDs for comparison)
    f1 = f1_score(true_token_ids, predicted_token_ids.cpu().numpy(), average='micro')
    
    return f1



size = '1M'

# Load test dataset
file_path = 'data/generated_polymer_smiles_dev.txt'

with open(file_path, 'r') as file:
    psmiles_strings = [line.strip() for line in file]
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained('original_tok')
model = DebertaV2ForMaskedLM.from_pretrained('original_model').to(device)
f1_original = evaluate(tokenizer, model, psmiles_strings, device)

tokenizer = DebertaV2Tokenizer(f"spm_{size}.model",f"spm_{size}.vocab")
model = DebertaV2ForMaskedLM.from_pretrained(f'model_{size}_final/').to(device)
f1_1M = evaluate(tokenizer, model, psmiles_strings, device)

print(f1_original, f1_1M)

result = {
    'pretrain size': ['90M (original)', '1M'],
    'f1 score': [f1_original, f1_1M]
}
csv_file = "masking_evaluation.csv"

# Writing to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data.keys())
    
    # Write headers
    writer.writeheader()
    # Write rows by zipping the lists in the dictionary
    for row in zip(*data.values()):
        writer.writerow(dict(zip(data.keys(), row)))