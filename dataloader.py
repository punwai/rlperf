import json
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from config import DatasetConfig
import pyarrow.parquet as pq

def create_dataloaders(config: DatasetConfig, transform=None):
    """
    Create train and test DataLoaders from configuration.
    
    Args:
        config: DataConfig object with dataset paths and parameters
        transform: Optional transform to apply to the data
    
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    # Create datasets
    train_dataset = StasisDataset(
        filepath=f"{config.dataset_path}/train.parquet",
        transform=transform,
    )
    
    test_dataset = StasisDataset(
        filepath=f"{config.dataset_path}/test.parquet",
        transform=transform,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    return train_loader, test_loader

class StasisDataset(Dataset):
    def __init__(self, filepath: str, transform=None):
        self.data = pq.read_table(filepath).to_pandas()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        item = self.data.iloc[index].to_dict()
        item["prompt"] = json.loads(item["prompt"])
        return item