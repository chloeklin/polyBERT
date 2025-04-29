import time
import os
import argparse
import logging
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling
)

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from utils.model_utils import load_tokenizer, load_model
from utils.data_utils import load_pretrain_data


"""Global variables"""
tokeniser = None

class TimingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")


def main():
    global tokeniser
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--root_dir', type=str, help='Root directory of repository')
    parser.add_argument('--size', type=str, help='Pretraining size')
    parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    ngpus=args.ngpus
    logging.basicConfig(level=logging.INFO)
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)
    
    """Tokeniser"""
    tokeniser = load_tokenizer(f"{args.root_dir}/tokeniser/spm/", size)
    
    logging.info('Init tokeniser')

    """Model"""
    model = load_model(tokeniser, 'lightning')
    logging.info('Init model')

    """Dataset"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser, mlm=True, mlm_probability=0.15
    )
    
    dataset_train, dataset_test = load_pretrain_data(os.path.join(args.root_dir,"data","pretrain",f"tokenised_{size}"))
    train_loader = DataLoader(dataset_train, batch_size=60, shuffle=True, collate_fn=data_collator, num_workers=11)
    test_loader = DataLoader(dataset_test, batch_size=60, shuffle=False, collate_fn=data_collator, num_workers=11)
    logging.info('Setup datasets')
    
    """Train model"""
    timing_callback = TimingCallback() 
    ckpt_callback = ModelCheckpoint(dirpath=os.path.join(args.root_dir, "pretrain_models",f"model_{size}_ds"), save_top_k=1, save_last=True, monitor="val_loss", mode="min", every_n_train_steps=5_000)

    last_ckpt_path = os.path.join(args.root_dir, "pretrain_models",f"model_{size}_ds","last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None

    trainer = Trainer(deterministic=True,
                      default_root_dir=os.path.join(args.root_dir, "pretrain_models",f"model_{size}_ds"),
                      max_epochs=2,
                      accelerator='gpu',
                      devices=ngpus,
                      strategy="deepspeed_stage_2",
                      precision=16,
                      log_every_n_steps=1_000,
                      callbacks=[ckpt_callback,timing_callback],
                      resume_from_checkpoint=resume_ckpt
    )
    trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    logging.info('Init trainer')
    
    start = time.process_time()
    trainer.fit(model, train_loader, test_loader,)#ckpt_path=f"./model_{size}_ds/epoch=1-step=210000.ckpt"
    end = time.process_time()
    elapsed = end - start
    print(f"elapsed time: {elapsed}")

    file_path = os.path.join(args.root_dir, 'pretraining_info.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path).astype(object)
    else:
        df = pd.DataFrame(columns=['pretrain size', f'model train time ({ngpus} GPUs) [ds]'])
    
    if size in df['pretrain size'].values:
        df.loc[df['pretrain size'] == size, f'model train time ({ngpus} GPUs) [ds]'] = elapsed
    else:
        df = df.append({'pretrain size': size, f'model train time ({ngpus} GPUs) [ds]': elapsed}, ignore_index=True)

    df.to_csv(file_path, index=False)

    
if __name__ == '__main__':
    main()
