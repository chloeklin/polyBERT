import os
import torch
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2Tokenizer
from model.polyBERT import polyBERT
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def load_tokenizer(model_dir, size):
    return DebertaV2Tokenizer(f"{model_dir}/spm_{size}.model", f"{model_dir}/spm_{size}.vocab")

def load_model(tokeniser, mode, state_path=None, config=None, ckpt=None):
    if config is None:
        config = DebertaV2Config(
            vocab_size=265,
            hidden_size=600,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=512,
            pad_token_id=3
        )
    if mode == 'torch':
        model = DebertaV2ForMaskedLM(config)
        model.resize_token_embeddings(len(tokeniser))
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
    else:
        if ckpt:
            model = polyBERT.load_from_checkpoint(ckpt)
        else:
            model = polyBERT(config,tokeniser)
    return model

def zero_checkpoint_to_fp32(file_path, size):
    """
    To process checkpoints to state dict, do:
        fp32_to_state_dict("50M")
    """
    fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(f'{file_path}/model_{size}_ds/last.ckpt')
    # Remove the "model." prefix from each key in the state dictionary
    updated_state_dict = {key.replace("model.", ""): value for key, value in fp32_state_dict.items()}
    os.makedirs(f"{file_path}/model_state_dict/", exist_ok=True)
    torch.save(updated_state_dict, f"{file_path}/model_state_dict/model_{size}_state_dict.pth")

