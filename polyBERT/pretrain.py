import time
import torch
import argparse
import logging
import numpy as np
import random
import pandas as pd
from datasets import Dataset
from torch.nn import DataParallel
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
# import lightning as L


print(f"done importing modules!")

def get_unwrapped_model(model):
    return model.module if hasattr(model, 'module') else model

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    parser.add_argument('--ndevices', type=str, help='Number of CPU/GPUs')
    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    ndevices=args.ndevices

    # Set seed for Python's built-in random module
    random.seed(1)

    # Set seed for NumPy
    np.random.seed(1)

    # Set seed for PyTorch CPU
    torch.manual_seed(1)
    torch.set_num_threads(int(ndevices))

    
    """Device"""
    device = torch.device('cpu')
    
    """ Tokeniser"""
    tokenizer = DebertaV2Tokenizer(f"spm/spm_{size}.model",f"spm/spm_{size}.vocab")
    logging.basicConfig(level=logging.INFO)
    
    config = DebertaV2Config(vocab_size=265, 
                          hidden_size=600,
                          num_attention_heads=12,
                          num_hidden_layers=12,
                          intermediate_size=512,
                          pad_token_id=3
                          )

    print(f"loaded tokeniser!")


    """ Model"""
    model = DebertaV2ForMaskedLM(config=config).to(device)
    
    # Resize token embedding to tokenizer
    model.resize_token_embeddings(len(tokenizer))

    print(f"loaded model!")

    """Dataset"""
    dataset_train = Dataset.load_from_disk(f"data/tokenized_{size}/train")
    dataset_test = Dataset.load_from_disk(f"data/tokenized_{size}/test")
    
    dataset_train.set_format(type='torch', columns=['input_ids'])
    dataset_test.set_format(type='torch', columns=['input_ids'])
        
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print(f"loaded dataset!")


    """Trainer"""
    training_args = TrainingArguments(
    output_dir=f"./model_{size}_cpu/",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1000, #30
    per_device_eval_batch_size=1000, #30
    save_steps=5_0000,
    save_total_limit=1,
    fp16=False,
    logging_steps=1_0000,
    prediction_loss_only=True,
    # disable_tqdm=True,
    dataloader_num_workers=16
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
    )
    print(f"trainer setup!")


    """Training"""
    # read pretrain file
    file_path = 'pretrain_info.csv'
    df = pd.read_csv(file_path)
    df = df.astype(object)
        
    start = time.process_time()
    a = trainer.train(resume_from_checkpoint=False) #trainer.train() #
    end = time.process_time()
    cpu_time = end - start
    df.loc[df['pretrain size'] == size, f'model train time ({ndevices} CPUs)'] = cpu_time

  
    df.to_csv('pretrain_info.csv', index=False)
    
    trainer.save_model(f"./model_{size}_final/")

if __name__ == '__main__':
    main()
