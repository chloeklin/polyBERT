"""Utility packages"""
import os
import csv
import random
import argparse
"""Model setup"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
"""Evaluation"""
from sklearn.metrics import f1_score
"""Deepspeed"""
import lightning as L
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


class polyBERT(L.LightningModule):
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
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        val_loss = outputs.loss
        # Log the validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = FusedAdam(self.parameters(), lr=5e-5)
        return optimizer
    
    
    def predict_step(self, batch, batch_idx):        
        # Run inference to get predictions for masked tokens
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = outputs.logits

        # Get the predicted token IDs for the masked positions
        masked_indices = (batch['input_ids'] == self.tokeniser.mask_token_id).nonzero(as_tuple=True)
        predicted_token_ids = predictions[masked_indices].argmax(dim=-1)
        
        # Return predicted token IDs
        return predicted_token_ids.cpu().numpy()
    

def evaluate(size, psmiles_strings, batch_size, ngpus, csv_file):
    # save_path = f"./model_{size}_ds/last.ckpt"
    output_path = f"./model_{size}_ds/last.pt"
    # convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)


    # Init tokeniser and model
    tokeniser = DebertaV2Tokenizer(f"spm/spm_{size}.model",f"spm/spm_{size}.vocab")
    model = polyBERT.load_from_checkpoint(output_path)
    
    # Mask 15% of tokens of each string in test data
    masked_psmiles, ground_truth = create_masked_test_set(tokeniser,psmiles_strings)
    
    # Tokenize the sentences
    inputs = tokeniser(masked_psmiles, return_tensors='pt', padding=True)
    
    # Create a DataLoader to batch inputs
    dataset = Dataset.load_from_disk(f"data/tokenized_{size}/train")
    
    dataset.set_format(type='torch', columns=['input_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=11)

    
    all_predicted_token_ids = []
    all_true_token_ids = tokeniser.convert_tokens_to_ids(ground_truth)
    
    trainer = Trainer(deterministic=True,
                      accelerator='gpu',
                      devices=ngpus,
                      strategy="deepspeed_stage_3",
                      enable_progress_bar=True,
                      precision=16)
    predictions = trainer.predict(model, dataloader)

    # Concatenate predictions
    for batch_predictions in predictions:
        all_predicted_token_ids.append(batch_predictions)

    # Compute F1 score (using token IDs for comparison)
    f1 = f1_score(all_true_token_ids, all_predicted_token_ids, average='micro')
    print(f"F1 score: {f1}")

    write_row_to_csv(csv_file, [size,f1])




def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--size', type=str, help='Pretraining size')
    parser.add_argument('--ngpus', type=int, help='Number of GPUs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')

    # Parse the arguments
    args = parser.parse_args()
    size=args.size
    ngpus=args.ngpus
    batch=args.batch
    
    # sets seeds for numpy, torch and python.random.
    seed_everything(1, workers=True)

    # Load test dataset
    file_path = 'data/generated_polymer_smiles_dev.txt'
    csv_file = "masking_evaluation.csv"


    with open(file_path, 'r') as file:
        psmiles_strings = [line.strip() for line in file]

    # Load tokenizer and model
    evaluate(size, psmiles_strings, batch, ngpus, csv_file)


    # tokeniser = DebertaV2Tokenizer.from_pretrained('original_tok')
    # model = DebertaV2ForMaskedLM.from_pretrained('original_model')
    # f1_original = evaluate(model, tokeniser, psmiles_strings)
    # write_row_to_csv(csv_file, ['original(90M)',f1_original])

if __name__ == '__main__':
    main()