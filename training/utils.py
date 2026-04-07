import torch
from torch import Tensor


def get_non_diag_values(matrix: Tensor) -> Tensor:
    # works for [N, N, C] matrices
    non_diag_idx = ~torch.eye(matrix.shape[0], dtype=torch.bool)
    return matrix[non_diag_idx]


def cannot_compare_instances(matrix: Tensor) -> bool:
    """
    Returns a boolean that indicates if the matrix has less
    than 2 instances. Order matrix must compare at least 2
    objects.
    """
    return matrix.shape[0] < 2
