from math import ceil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

class RegularGridInterpolator(nn.Module):
    def __init__(self, points, values, method='linear'):
        """
        points: list of 1D tensors defining the regular grid
        values: tensor with shape (grid_dims..., *value_shape) where value_shape can be multi-dimensional
                For example, for a 3D grid with 3-channel vectors: (D, H, W, 3)
        """
        super().__init__() # Call the nn.Module constructor
        
        self.method = method
        self.ndim = len(points)
        
        # Check that value shape matches grid
        expected_shape = tuple(len(p) for p in points)
        if values.shape[:len(expected_shape)] != expected_shape:
            raise ValueError(
                f"Values tensor shape {values.shape} does not "
                f"match grid dimensions {expected_shape}"
            )
        
        # Extract value shape (extra dimensions beyond grid dimensions)
        self.value_shape = values.shape[len(expected_shape):]
        self.n_value_dims = len(self.value_shape)
        
        # --- Register buffers ---
        
        # 1. Store the actual grid points for each dimension
        # This is necessary for handling uneven grid spacing
        grid_points = [torch.as_tensor(p, dtype=torch.float32) for p in points]
        for i, gp in enumerate(grid_points):
            self.register_buffer(f"grid_points_{i}", gp.contiguous())
        
        # Store grid sizes for convenience
        self.grid_sizes = [len(p) for p in points]
        
        # 2. Store the values tensor, pre-shaped for grid_sample
        # grid_sample expects (N, C, D, H, W) for 3D or (N, C, H, W) for 2D
        # We need to move channel dimensions to the second position
        if self.ndim == 2:
            # values: (H, W, *value_shape) -> (1, *value_shape, H, W)
            # Flatten value_shape into a single channel dimension
            if self.n_value_dims > 0:
                value_size = int(np.prod(self.value_shape))
                values_reshaped = values.view(*expected_shape, value_size)
                values_reshaped = values_reshaped.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            else:
                # Scalar values
                values_reshaped = values.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif self.ndim == 3:
            # values: (D, H, W, *value_shape) -> (1, *value_shape, D, H, W)
            # Flatten value_shape into a single channel dimension
            if self.n_value_dims > 0:
                value_size = int(np.prod(self.value_shape))
                values_reshaped = values.view(*expected_shape, value_size)
                values_reshaped = values_reshaped.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
            else:
                # Scalar values
                values_reshaped = values.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        else:
            raise ValueError(f"{self.ndim}D interpolation not supported")
            
        self.register_buffer("values_reshaped", values_reshaped)

    def forward(self, xi):
        """
        xi: query points, shape (N, ndim)
        Returns: interpolated values, shape (N, *value_shape)
        
        The nn.Module's __call__ method will automatically call this `forward` method.
        """
        
        # 1. Convert query coordinates to normalized coordinates [-1, 1]
        # For uneven spacing, we need to find the grid cell for each coordinate
        # and compute the normalized position based on actual grid point positions
        normalized_list = []
        
        for dim in range(self.ndim):
            # Get grid points for this dimension
            grid_points = getattr(self, f"grid_points_{dim}")
            n_grid = self.grid_sizes[dim]
            
            # Query coordinates for this dimension
            x = xi[:, dim]
            
            # Clamp to grid bounds (for border padding)
            x_min = grid_points[0]
            x_max = grid_points[-1]
            x_clamped = torch.clamp(x, x_min, x_max)
            
            # Find grid indices using searchsorted
            # We want to find the cell [grid_points[i], grid_points[i+1]] that contains x_clamped
            # searchsorted with right=True returns the rightmost insertion point
            # For x in [grid_points[i], grid_points[i+1]), it returns i+1
            idx_high = torch.searchsorted(grid_points, x_clamped, right=True)
            
            # Handle edge cases
            # Since x_clamped is clamped to [grid_points[0], grid_points[-1]]:
            # - idx_high will be in [1, n_grid]
            # - We want idx_high in [1, n_grid-1] for valid cells
            idx_high = torch.clamp(idx_high, 1, n_grid - 1)
            idx_low = idx_high - 1
            
            # Get the actual grid point values
            p_low = grid_points[idx_low]
            p_high = grid_points[idx_high]
            
            # Compute fractional position within the cell
            # Handle division by zero (when p_high == p_low, which shouldn't happen but be safe)
            cell_size = p_high - p_low
            alpha = torch.where(
                cell_size > 1e-10,
                (x_clamped - p_low) / cell_size,
                torch.zeros_like(x_clamped)
            )
            
            # Convert to normalized coordinates [-1, 1]
            # grid_sample with align_corners=True maps:
            # -1 to grid index 0, +1 to grid index (n_grid - 1)
            # So grid index i maps to: -1 + 2 * i / (n_grid - 1)
            normalized_idx = idx_low.float() + alpha
            normalized_coord = -1.0 + 2.0 * normalized_idx / (n_grid - 1.0)
            
            normalized_list.append(normalized_coord)
        
        # Stack normalized coordinates
        normalized = torch.stack(normalized_list, dim=1)
        
        # 2. Flip coordinates for grid_sample
        # (x_norm, y_norm, z_norm) -> (z_norm, y_norm, x_norm)
        grid_coords = normalized.flip(-1)
        
        # 3. Reshape grid for grid_sample
        if self.ndim == 2:
            # (N, 2) -> (1, N_points, 1, 2)
            grid = grid_coords.view(1, -1, 1, self.ndim)
            mode = 'bilinear' if self.method == 'linear' else 'nearest'
        elif self.ndim == 3:
            # (N, 3) -> (1, N_points, 1, 1, 3)
            grid = grid_coords.view(1, -1, 1, 1, self.ndim)
            mode = 'bilinear' if self.method == 'linear' else 'nearest'
        
        # 4. Interpolate
        # self.values_reshaped is already on the correct device
        # Result shape: (1, C, N_points, 1, 1) for 3D or (1, C, N_points, 1) for 2D
        result = F.grid_sample(
            self.values_reshaped, 
            grid, 
            mode=mode, 
            align_corners=True, 
            padding_mode='border'
        )
        
        # 5. Reshape result back to (N_points, *value_shape)
        # Remove batch dimension and spatial dimensions, then reshape channels
        if self.ndim == 2:
            # result: (1, C, N_points, 1) -> (N_points, C) or (N_points,)
            result = result.squeeze(0).squeeze(-1).permute(1, 0)  # (N_points, C)
        elif self.ndim == 3:
            # result: (1, C, N_points, 1, 1) -> (N_points, C) or (N_points,)
            result = result.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0)  # (N_points, C)
        
        # Reshape to restore original value_shape
        if self.n_value_dims > 0:
            # Multi-dimensional values: reshape to (N_points, *value_shape)
            n_points = result.shape[0]
            result = result.view(n_points, *self.value_shape)
        else:
            # Scalar values: squeeze out the channel dimension to get (N_points,)
            result = result.squeeze(-1)
        
        return result

def trunc_Zd(n, d=3, flatten=True, dtype=torch.float):
    """
    Returns [-n, n]^d with shape [2n+1, 2n+1, ..., 2n+1, d]
    """
    # Create ranges for each dimension
    ranges = [torch.arange(-n, n + 1, dtype=dtype) for _ in range(d)]
    
    # Create the meshgrid
    grid = torch.meshgrid(*ranges, indexing='ij')
    
    # Stack the grids along a new dimension
    grid = torch.stack(grid, dim=-1)
    
    if flatten:
        # Reshape to (num_points, d)
        return grid.reshape(-1, d)
    
    return grid

def interpolator(f, points, device, bs=None, verbose=False):
    """
    Wrapper for `RegularGridInterpolator`
    """
    dim = len(points)
    grid = torch.stack(torch.meshgrid(*points, indexing="ij"), dim=-1)
    grid_shape = grid.shape
    grid = grid.view(-1, dim)
    n_batch = ceil(grid.shape[0] / bs)
    result = []
    for batch_i in trange(n_batch, disable=not verbose, desc="Creating interpolator"):
        batch = grid[batch_i * bs : (batch_i + 1) * bs]
        result.append(f(batch))
    values = torch.cat(result, 0).view(*grid_shape[:-1], -1)
    interpolator = RegularGridInterpolator(points, values).to(device)
    return interpolator, values, grid


def periodic_stationary_interpolator(
    f, dim, density, device, bs=128, exponent=1, verbose=False, return_values=False
):
    """
    Create an interpolating object for a periodic stationary kernel with period [0, 2pi]
    Uses grid values of (linspace(-1, 1, density) ** exponent * jnp.pi)^dim
    """
    xis = (torch.linspace(-1, 1, density, device=device) ** exponent) * torch.pi
    f_interpolator, values, points = interpolator(f, (xis,) * dim, device, bs, verbose)

    def helper(x1, x2):
        x_diff = x1 - x2
        x_diff = (x_diff + torch.pi) % (2 * torch.pi) - torch.pi
        return f_interpolator(x_diff).squeeze()

    # Match JAX API: return f_interpolator.values instead of separate values
    return helper if not return_values else (helper, values, points)