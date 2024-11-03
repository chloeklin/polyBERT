from evaluate import *

size="1M"
# ngpus=args.ngpus
batch=3

# sets seeds for numpy, torch and python.random.
seed_everything(1, workers=True)
logging.basicConfig(level=logging.INFO)


# Load test dataset
file_path = 'data/generated_polymer_smiles_dev.txt'


with open(file_path, 'r') as file:
    psmiles_strings = [line.strip() for line in file]

# Load tokenizer and model
evaluate(size, psmiles_strings[:10], batch)




# import datasets
# import argparse
# from transformers import DebertaV2Tokenizer
# from datasets import load_dataset



# size="1M"
# original_pretrain_file = 'generated_polymer_smiles_train'

# dataset = load_dataset('text', data_files={'train': f'data/{original_pretrain_file}_{size}.txt',
#                             'test': 'data/generated_polymer_smiles_dev.txt'},
#                             cache_dir=f'dataset_cache_{size}/')