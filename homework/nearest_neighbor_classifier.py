import torch

class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        """
        self.data, self.label = self.make_data(x, y)
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        self.data_normalized = self.input_normalization(self.data)
    
    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert lists to PyTorch tensors safely, ensuring no gradient tracking issues.
        """
        return torch.tensor(x, dtype=torch.float32).clone().detach(), torch.tensor(y, dtype=torch.float32).clone().detach()



    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mean and standard deviation along columns.
        """
        return x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True)

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input data using stored mean and std.
        """
        return (x - self.data_mean) / self.data_std

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the closest data point using Euclidean distance.
        """
        x = self.input_normalization(x)
        distances = torch.norm(self.data_normalized - x, dim=1)
        idx = torch.argmin(distances)  # Get index of minimum distance
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get k-nearest neighbors using Euclidean distance.
        """
        x = self.input_normalization(x)
        distances = torch.norm(self.data_normalized - x, dim=1)
        idx = torch.topk(distances, k, largest=False).indices  # Get indices of k smallest distances
        return self.data[idx], self.label[idx]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Predict regression output as the average of k-nearest neighbors' labels.
        """
        _, labels = self.get_k_nearest_neighbor(x, k)
        return labels.mean()
