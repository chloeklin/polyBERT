import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(df: pd.DataFrame, filename: str, val_size=0.1, test_size=0.1):
    train_data, temp_data = train_test_split(df, test_size=(val_size+test_size), random_state=1)
    val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=1)

    # Save the splits to new CSV files
    train_data.to_csv(f'../regression_data/train_{filename}.csv', index=False)
    val_data.to_csv(f'../regression_data/val_{filename}.csv', index=False)
    test_data.to_csv(f'../regression_data/test_{filename}.csv', index=False)