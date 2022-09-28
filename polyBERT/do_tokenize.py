import datasets
from transformers import DebertaV2Tokenizer
from datasets import load_dataset

# Loads the tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("./")

dataset = load_dataset('text', data_files={'train': 'generated_smiles_train.txt',
                             'test': 'generated_smiles_dev.txt'},
                             cache_dir='dataset_cache/')

def tokenize(data):
    res = tokenizer(data['text'])
    return {'input_ids': res['input_ids']}

dataset = dataset.map(tokenize, batched=True, batch_size=10_000, num_proc=10) 
dataset.save_to_disk('dataset_tokenized_all')

