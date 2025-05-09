import os
import argparse
from transformers import DebertaV2Tokenizer
from datasets import load_dataset
from train_tokenizer import train

"""Global vairables"""
tokenizer = None

def tokenize(data):
    global tokenizer
    res = tokenizer(data['text'])
    return {'input_ids': res['input_ids']}

def load_or_train_tokenizer(root_dir, size):
    model_path = f"{root_dir}/tokeniser/spm_{size}/" #spm/spm_{size}.model
    try:
        print("Trying to load existing tokenizer...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Tokenizer not found. Training new tokenizer")
        train(root_dir, size)
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        print("Training done")
    return tokenizer

def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='Root directory of repository')
    parser.add_argument('--size', type=str, help='Pretraining size')
    # Parse the arguments
    args = parser.parse_args()

    tokenizer = load_or_train_tokenizer(args.root_dir, args.size)

    dataset = load_dataset('text', 
                           data_files={
                            'train': os.path.join(args.root_dir, "data","pretrain", f"psmiles_train_{args.size}.txt"),
                            'test': os.path.join(args.root_dir, "data","pretrain", "test.txt")})
    print(f"Tokenising dataset: {os.path.join(args.root_dir, "data","pretrain", f"psmiles_train_{args.size}.txt")}")
    dataset = dataset.map(tokenize, batched=True, batch_size=10_000, num_proc=10) 
    dataset.save_to_disk(f'{args.root_dir}/data/pretrain/tokenised_{args.size}')
    print(f'Tokenised dataset saved to {args.root_dir}/data/pretrain/tokenised_{args.size}')

if __name__ == '__main__':
    main()
