import torch


class PyTorchBasics:
    """
    Implement the following python code with PyTorch.
    Use PyTorch functions to make your solution efficient and differentiable.

    General Rules:
    - No loops, no function calls (except for torch functions), no if statements
    - No numpy
    - PyTorch and tensor operations only
    - No assignments to results x[1] = 5; return x
    - A solution requires less than 10 PyTorch commands

    The grader will convert your solution to torchscript and make sure it does not
    use any unsupported operations (loops etc).
    """

    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        # from the tensor select every 3rd element 
        # used slicing 
        return x[::3]
    
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
        # Calculate maximum & last dimension
        # Return only the values
        return x.max(dim=-1).values
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        # Extract unique values and then return sorted order
        return torch.unique(x, sorted=True)
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # calculate mean of x and then count elements in y which are more than mean
        return (y > x.mean()).sum()
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        # Return the transpose of the tensor
        return x.T
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        # Extract and return the main diagonal elements
        return torch.diag(x)
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        # Flip the tensor left-to-right and then extract its diagonal
        return torch.diag(torch.fliplr(x))
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        # Compute the cumulative sum along the only dimension (dim=0)
        return torch.cumsum(x, dim=0)
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        # Compute cumulative sum along rows first, then along columns
        return torch.cumsum(torch.cumsum(x, dim=0), dim=1)
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Replace elements in x that are less than c with 0, keeping other elements unchanged
        return torch.where(x < c, torch.tensor(0.0, dtype=x.dtype, device=x.device), x)
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Find indices  where elements in x are less than c and return them as a 2 x n tensor
        return torch.nonzero(x < c, as_tuple=False).T
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        # Use boolean indexing to extract elements from x where mask m is True.
        return x[m]
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Concatenate x and y, then calculate differences between consecutive elements
        return torch.diff(torch.cat((x, y)))
        raise NotImplementedError

    @staticmethod
    def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # calculate absolute differences between each x and all y, check if any are < 1e-3, and count them
        return ((x.unsqueeze(1) - y).abs() < 1e-3).any(dim=1).sum()
        raise NotImplementedError
