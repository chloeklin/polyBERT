import lightning as L
from transformers import DebertaV2ForMaskedLM, DataCollatorForLanguageModeling
from deepspeed.ops.adam import FusedAdam

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

    
    def predict_step(self, batch, batch_idx):        
        # Run inference to get predictions for masked tokens
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        predictions = outputs.logits

        # Get the predicted token IDs for the masked positions
        masked_indices = (batch['input_ids'] == self.tokeniser.mask_token_id).nonzero(as_tuple=True)
        predicted_token_ids = predictions[masked_indices].argmax(dim=-1)
        
        # Return predicted token IDs
        return predicted_token_ids.cpu().numpy()
    
    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = FusedAdam(self.parameters(), lr=5e-5)
        return optimizer
    