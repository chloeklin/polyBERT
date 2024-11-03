import pandas as pd
from sklearn.model_selection import train_test_split
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import torch

def train_val_test_split(df: pd.DataFrame, filename: str, val_size=0.1, test_size=0.1):
    train_data, temp_data = train_test_split(df, test_size=(val_size+test_size), random_state=1)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=1)

    # Save the splits to new CSV files
    train_data.to_csv(f'../regression_data/train_{filename}.csv', index=False)
    val_data.to_csv(f'../regression_data/val_{filename}.csv', index=False)
    test_data.to_csv(f'../regression_data/test_{filename}.csv', index=False)
    
def fp32_to_state_dict(size):
    fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(f'model_{size}_ds/last.ckpt')
    # Remove the "model." prefix from each key in the state dictionary
    updated_state_dict = {key.replace("model.", ""): value for key, value in fp32_state_dict.items()}
    torch.save(updated_state_dict, f"model_{size}_state_dict.pth")