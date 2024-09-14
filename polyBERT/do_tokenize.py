import datasets
import argparse
from transformers import DebertaV2Tokenizer
from datasets import load_dataset

original_pretrain_file = 'generated_polymer_smiles_train'


def tokenize(data):
    res = tokenizer(data['text'])
    return {'input_ids': res['input_ids']}

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    # Parse the arguments
    args = parser.parse_args()
    size=args.size

    # Loads the tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(f"./spm_{size}.model")
    dataset = load_dataset('text', data_files={'train': f'{original_pretrain_file}_{size}.txt',
                                'test': 'generated_polymer_smiles_dev.txt'},
                                cache_dir=f'dataset_cache_{size}/')

    dataset = dataset.map(tokenize, batched=True, batch_size=10_000, num_proc=10) 
    dataset.save_to_disk(f'dataset_tokenized_{size}')

if __name__ == '__main__':
    main()
