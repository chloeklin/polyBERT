"""Utility packages"""
import time
import torch
import argparse
import logging
import pandas as pd
"""Model setup"""
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
"""Deepspeed"""
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.optim import Adam
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


class TimingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")

class DebertaMLM(L.LightningModule):
    def __init__(self, config, tokeniser):
        super().__init__()
        self.tokeniser = tokeniser
        self.model = DebertaV2ForMaskedLM(config=config)
        self.model.resize_token_embeddings(len(tokeniser))
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokeniser, mlm=True, mlm_probability=0.15
        )
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], logger=True)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], logger=True)
        val_loss = outputs.loss
        # Log the validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=5e-5)
        return optimizer
    

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    ngpus=args.ngpus
    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)

    """Pretraining time"""
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    
    """Tokeniser"""
    tokeniser = DebertaV2Tokenizer(f"spm_{size}.model",f"spm_{size}.vocab")
    logging.basicConfig(level=logging.INFO)
    logging.info('Init tokeniser')

    """Model"""
    # Configuration for the DeBERTa model
    config = DebertaV2Config(
        vocab_size=265,
        hidden_size=600,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=512,
        pad_token_id=3
    )
        
    model = DebertaMLM(config,tokeniser)
    logging.info('Init model')

    """Dataset"""
    dataset_train = Dataset.load_from_disk(f"data/tokenized_{size}/train")
    dataset_test = Dataset.load_from_disk(f"data/tokenized_{size}/test")
    
    dataset_train.set_format(type='torch', columns=['input_ids'])
    dataset_test.set_format(type='torch', columns=['input_ids'])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser, mlm=True, mlm_probability=0.15
    )

    train_loader = DataLoader(dataset_train, batch_size=30, shuffle=True, collate_fn=data_collator, num_workers=11)
    test_loader = DataLoader(dataset_test, batch_size=30, shuffle=False, collate_fn=data_collator, num_workers=11)
    logging.info('Setup datasets')
    
    """Train model"""
    timing_callback = TimingCallback()
    trainer = Trainer(deterministic=True,
                      default_root_dir=f"./model_{size}/",
                      max_epochs=2,
                      accelerator='gpu',
                      devices=ngpus,
                      strategy=DeepSpeedStrategy(config="deepspeed_config.json"),
                      precision=16,
                      log_every_n_steps=1_000,
                      callbacks=[ModelCheckpoint(dirpath=f"./model_{size}/", save_top_k=1, save_last=True, monitor="train_loss", every_n_train_steps=5_000),timing_callback]
    )
    # trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False #turn this off
    logging.info('Init trainer')
    

    # file_path = 'pretrain_info.csv'
    # df = pd.read_csv(file_path)
    # start_event.record()
    trainer.fit(model, train_loader, test_loader)
    # end_event.record()
    # torch.cuda.synchronize()
    # gpu_time = start_event.elapsed_time(end_event) / 1000  # Time in milliseconds
    # df = df.astype(object)
    # df.loc[df['pretrain size'] == size, f'model train time ({ngpus} GPUs) [ds]'] = gpu_time
    # df.to_csv('pretrain_info.csv', index=False)
    
    # trainer.save_model(f"./model_{size}_final/")

if __name__ == '__main__':
    main()
