from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class AbstractKernel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x1, x2, diag=False) -> torch.Tensor:
        pass

    @abstractmethod
    def spectral_density(self, n) -> torch.Tensor:
        pass

def matern32_helper(x1, x2, lengthscale) -> torch.Tensor:
    rt3 = 3 ** (1 / 2)
    w = rt3 * (torch.abs(x1 - x2) - (1 / 2))
    w_star = rt3 / 2

    cosh_term = torch.exp((w - w_star) / lengthscale) + torch.exp(
        (-w - w_star) / lengthscale
    )
    first = 2 * lengthscale + rt3 / torch.tanh(rt3 / (2 * lengthscale))
    second = 2 * w * torch.tanh(w / lengthscale)

    return cosh_term * (torch.pi**2 * lengthscale / 3) * (first - second)

def matern32_kernel_s1(x1, x2, lengthscale, variance) -> torch.Tensor:
    """
    Matern 3/2 kernel on S^1 as defined in https://arxiv.org/abs/2006.10160
    """
    return (
        variance
        * matern32_helper(x1 / (2 * torch.pi), x2 / (2 * torch.pi), lengthscale)
        / matern32_helper(torch.zeros_like(lengthscale), torch.zeros_like(lengthscale), lengthscale)
    )

def matern32_kernel_td(x1, x2, lengthscale, variance) -> torch.Tensor:
    """
    T^d = S^1 x ... x S^1 (d times)
    """
    kernel_per_dim = matern32_kernel_s1(x1, x2, lengthscale, variance)
    return torch.prod(kernel_per_dim)

def matern32_spectral_density_s1(n, lengthscale, variance) -> torch.Tensor:
    rt3 = 3**0.5
    second = (3 / (lengthscale**2) + 4 * torch.pi**2 * n**2) ** (-2)
    first = (
        2
        * rt3
        * torch.tanh(rt3 / (2 * lengthscale))
        / ((2 * torch.pi) ** (-2) * lengthscale)
    )

    result = first * second
    result /= matern32_helper(torch.zeros_like(lengthscale), torch.zeros_like(lengthscale), lengthscale)
    return result * variance

def vectorized_matern32_kernel_td(x1, x2, lengthscale, variance) -> torch.Tensor:
    """
    Compute kernel matrix K[i,j] = matern32_kernel_td(x1[i], x2[j], lengthscale, variance)
    
    Args:
        x1: array of shape (n1, d)
        x2: array of shape (n2, d)
        lengthscale: array of shape (d,)
        variance: array of shape (d,)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Add broadcasting dimensions: x1 -> (n1, 1, d), x2 -> (1, n2, d)
    x1_expanded = x1[:, torch.newaxis, :]  # shape: (n1, 1, d)
    x2_expanded = x2[torch.newaxis, :, :]  # shape: (1, n2, d)
    
    # Compute kernels for all pairs and dimensions at once
    # This will broadcast to shape (n1, n2, d)
    kernel_per_dim = (
        variance
        * matern32_helper(
            x1_expanded / (2 * torch.pi), 
            x2_expanded / (2 * torch.pi), 
            lengthscale
        )
        / matern32_helper(
            torch.zeros_like(lengthscale), 
            torch.zeros_like(lengthscale), 
            lengthscale
        )
    )
    
    # Product over the last dimension to get final kernel matrix
    return torch.prod(kernel_per_dim, axis=-1)  # shape: (n1, n2)

def vectorized_matern32_kernel_diag(x1, x2, lengthscale, variance) -> torch.Tensor:
    """
    Compute diagonal elements: K[i] = matern32_kernel_td(x1[i], x2[i], lengthscale, variance)
    
    Args:
        x1: array of shape (n, d)
        x2: array of shape (n, d)
        lengthscale: array of shape (d,)
        variance: array of shape (d,)
    
    Returns:
        Diagonal kernel values of shape (n,)
    """
    # Compute kernels for all points and dimensions at once
    # x1 and x2 have shape (n, d), this will give shape (n, d)
    kernel_per_dim = (
        variance
        * matern32_helper(
            x1 / (2 * torch.pi), 
            x2 / (2 * torch.pi), 
            lengthscale
        )
        / matern32_helper(
            torch.zeros_like(lengthscale), 
            torch.zeros_like(lengthscale), 
            lengthscale
        )
    )
    
    # Product over the last dimension to get kernel values for each point
    return torch.prod(kernel_per_dim, axis=-1)  # shape: (n,)


class Matern32Kernel(AbstractKernel):
    """
    Stable version of Matern 3/2 kernel on T^d = S^1 x ... x S^1 (d times) as defined in https://arxiv.org/abs/2006.10160
    """
    def __init__(
        self,
        lengthscale=1.0, variance=1.0, dim=1
    ):
        super().__init__()
        self.dim = dim
        self.register_buffer("lengthscale", torch.tensor(lengthscale))
        self.register_buffer("variance", torch.tensor(variance))

    def forward(self, x1, x2, diag=False):
        """
        Compute the kernel matrix K[i,j] = matern32_kernel_td(x1[i], x2[j], lengthscale, variance)
        
        Args:
            x1: array of shape (n1, d)
            x2: array of shape (n2, d)
        
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        if diag:
            return vectorized_matern32_kernel_diag(x1, x2, self.lengthscale, self.variance)
        else:
            return vectorized_matern32_kernel_td(x1, x2, self.lengthscale, self.variance)
    
    def spectral_density(self, n):
        """
        Compute the spectral density of the Matern 3/2 kernel on T^d = S^1 x ... x S^1 (d times)
        
        Args:
            n: array of shape (n, d)
        
        Returns:
            Spectral density of shape (n,)
        """
        result = torch.ones(n.shape[0], device=n.device)
        for i in range(self.dim):
            result_dim = matern32_spectral_density_s1(n[:,i], self.lengthscale, self.variance)
            result *= result_dim
        return result