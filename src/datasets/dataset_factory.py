from .base_dataset import BaseDataset
from .wsi_dataset import WSIDataset

class DatasetFactory:
    _datasets = {
        "wsi": WSIDataset,
    }

    @staticmethod
    def create(
        dataset_name: str,
        **kwargs
    ) -> BaseDataset:
        """
        Factory method to create datasets
        
        Args:
            dataset_name: Type of dataset to create ("wsi").
            **kwargs: Arguments for the dataset. Can include both base class arguments and dataset-specific arguments.
        
        Returns:
            BaseDataset: Instance of the specified dataset
        
        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in DatasetFactory._datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DatasetFactory._datasets.keys())}")

        dataset_class = DatasetFactory._datasets[dataset_name]
        return dataset_class(**kwargs)
