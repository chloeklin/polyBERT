import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


class polyBERTDataset(Dataset):
    def __init__(self, dataframe, polyBERT):
        scaler = StandardScaler()
        embeddings = polyBERT.encode(dataframe['smiles'].to_list()) # Obtain polyBERT embeddings
        self.features = scaler.fit_transform(embeddings) # Normalise embedding
        self.targets = dataframe['value']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get the feature string and target value at the specified index
        embedding = self.features[idx]
        target = self.targets.iloc[idx]
        
        # Convert the arrays to to a torch tensor (assuming it's numerical)
        embedding = torch.tensor(embedding, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        
        return embedding, target
    
def load_dataset(data: str, polyBERT, batch_size=64):
    train_df = pd.read_csv(f"../regression_data/train_{data}.csv")
    test_df = pd.read_csv(f"../regression_data/val_{data}.csv")
    test_df = pd.read_csv(f"../regression_data/test_{data}.csv")

    train_loader = DataLoader(polyBERTDataset(train_df, polyBERT), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(polyBERTDataset(train_df, polyBERT), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(polyBERTDataset(test_df, polyBERT), batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader