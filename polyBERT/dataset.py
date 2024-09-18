import torch
from torch.utils.data import Dataset

class polyBERTDataset(Dataset):
    def __init__(self, dataframe, polyBERT):
        self.features = polyBERT.encode(dataframe['smiles'].to_list())
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