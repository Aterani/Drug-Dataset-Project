import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels, sensitive):
        self.data = torch.tensor(data)
        self.data = self.data.float()
        self.labels = torch.tensor(labels)
        self.labels = self.labels.float()
        self.sensitive = torch.tensor(sensitive)
        self.sensitive = self.sensitive.float()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        sensitive = self.sensitive[idx]
        return data, label, sensitive

