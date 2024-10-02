import time
import torch
import argparse
import logging
import pandas as pd
from datasets import Dataset
from torch.nn import DataParallel
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import lightning as L

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
    
    """Pretraining time"""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    """Device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.is_available() #checking if CUDA + Colab GPU works

    """ Tokeniser"""
    tokenizer = DebertaV2Tokenizer(f"spm_{size}.model",f"spm_{size}.vocab")
    logging.basicConfig(level=logging.INFO)
    
    config = DebertaV2Config(vocab_size=265, 
                          hidden_size=600,
                          num_attention_heads=12,
                          num_hidden_layers=12,
                          intermediate_size=512,
                          pad_token_id=3
                          )

    """ Model"""
    model = DebertaV2ForMaskedLM(config=config).to(device)
    
    # Resize token embedding to tokenizer
    model.resize_token_embeddings(len(tokenizer))


    """Dataset"""
    dataset_train = Dataset.load_from_disk(f"data/tokenized_{size}/train")
    dataset_test = Dataset.load_from_disk(f"data/tokenized_{size}/test")
    
    dataset_train.set_format(type='torch', columns=['input_ids'])
    dataset_test.set_format(type='torch', columns=['input_ids'])
        
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    """Trainer"""
    training_args = TrainingArguments(
        output_dir=f"./model_{size}_cpu/",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=30,
        per_device_eval_batch_size=30,
        save_steps=1_000, #5000
        save_total_limit=1,
        fp16=True,
        logging_steps=1_000,
        prediction_loss_only=True,
        # deepspeed = "deepspeed_config.json"
        # disable_tqdm=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
    )

    """Training"""
    # read pretrain file
    file_path = 'pretrain_info.csv'
    df = pd.read_csv(file_path)
    df = df.astype(object)
        
    if device=='cpu':
        start = time.process_time()
        a = trainer.train(resume_from_checkpoint=False) #trainer.train() #
        end = time.process_time()
        cpu_time = end - start
        df.loc[df['pretrain size'] == size, f'model train time ({ndevices} CPUs)'] = cpu_time

    else:

        start_event.record()
        a = trainer.train(resume_from_checkpoint=False) #trainer.train() #
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000  # Time in milliseconds
        df.loc[df['pretrain size'] == size, f'model train time ({ndevices} GPUs) [ds]'] = gpu_time
    df.to_csv('pretrain_info.csv', index=False)
    
    trainer.save_model(f"./model_{size}_final/")

if __name__ == '__main__':
    main()
