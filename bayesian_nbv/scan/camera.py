import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform


def intrinsic_matrix_from_fov(fov_deg, width, height, device='cuda'):
    """
    Create intrinsic matrix from field of view.
    
    Args:
        fov_deg: Field of view in degrees (horizontal)
        width: Image width
        height: Image height
    
    Returns:
        K: Intrinsic matrix (3, 3)
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assuming square pixels
    cx = width / 2
    cy = height / 2
    
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], device=device)
    
    return K

def fibonacci_sphere(
    N: int,
    radius: float = 1.0,
    device='cuda',
    dtype=torch.float32,
    random_oriented: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate N ~evenly distributed points on a sphere (radius `radius`).
    If `random_oriented` is True, applies a single global uniform random
    3D rotation so the sequence is randomly oriented (spacing preserved).

    Args:
      N: number of points (>=1)
      radius: sphere radius
      device, dtype: torch device/dtype for the output
      random_oriented: if True, rotate all points by a random SO(3)
      seed/generator: optional for reproducible random orientation

    Returns:
      pts: [N, 3] tensor
    """
    # --- base deterministic Fibonacci layout ---
    i = torch.arange(N, device=device, dtype=dtype)
    phi = (1.0 + 5.0**0.5) / 2.0  # golden ratio
    theta = 2.0 * torch.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / float(N)      # [-1, 1], equal-area bands
    r_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    pts = torch.stack((x, y, z), dim=-1) * radius

    if not random_oriented:
        return pts
    
    if generator is None and seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # --- random global rotation (Shoemake quaternion) ---
    def _random_rotation_matrix(_device, _dtype, _gen=None):
        u = torch.rand(3, device=_device, dtype=_dtype, generator=_gen)
        u1, u2, u3 = u[0], u[1], u[2]
        # unit quaternion (x,y,z,w)
        qx = torch.sqrt(1.0 - u1) * torch.sin(2.0 * torch.pi * u2)
        qy = torch.sqrt(1.0 - u1) * torch.cos(2.0 * torch.pi * u2)
        qz = torch.sqrt(u1)       * torch.sin(2.0 * torch.pi * u3)
        qw = torch.sqrt(u1)       * torch.cos(2.0 * torch.pi * u3)

        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        return torch.stack([
            torch.stack([1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)], dim=0),
            torch.stack([2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)], dim=0),
            torch.stack([2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)], dim=0),
        ], dim=0)

    if N == 1:
        # Single point: ensure uniform direction by rotating a pole
        R = _random_rotation_matrix(device, dtype, generator)
        base = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype) * radius
        return base @ R.T

    R = _random_rotation_matrix(device, dtype, generator)  # [3,3]
    return pts @ R.T


def cameras_on_sphere_lookat_origin(
    N: int,
    radius: float = 1.0,
    device='cuda',
    dtype=torch.float32,
    random_oriented: bool = False,
    seed: int | None = None,
    generator: torch.Generator | None = None,
):
    """
    Returns:
      R: [N, 3, 3] camera rotation matrices
      t: [N, 3] camera translations
      eye: [N, 3] camera centers (same as fibonacci_sphere output)
    """

    # 1) Camera positions (eye) on the sphere
    eye = fibonacci_sphere(N, radius=radius, device=device, dtype=dtype, random_oriented=random_oriented, seed=seed, generator=generator)

    # 2) Build per-camera 'up' that defaults to [0,1,0] but avoids colinearity with view dir.
    at = torch.zeros((N, 3), device=device, dtype=dtype)  # look-at origin
    default_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    alt_up     = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # View direction = (at - eye); only need direction, so normalize
    view_dir = (at - eye)
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-9)

    # Colinearity test: |dot(view_dir, up)| ~ 1  -> parallel/antiparallel
    dot_with_up = (view_dir * default_up).sum(dim=-1)
    parallel_mask = dot_with_up.abs() > (1.0 - 1e-6)

    up = default_up.repeat(N, 1)
    if parallel_mask.any():
        up[parallel_mask] = alt_up  # only switch for the degenerate cases (e.g., poles)

    # 3) Use PyTorch3D to get camera extrinsics
    R, t = look_at_view_transform(eye=eye, at=at, up=up, device=device)

    return R, t, eye


def cameras_looking_at_origin(
    eyes: torch.Tensor,
    device='cuda',
    dtype=torch.float32
):
    """
    eyes: (N, 3)
    """
    N = eyes.shape[0]

    with torch.no_grad():
        at = torch.zeros((N, 3), device=device, dtype=dtype)
        default_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        alt_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        view_dir = (at - eyes)
        view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-9)
        dot_with_up = (view_dir * default_up).sum(dim=-1)
        parallel_mask = dot_with_up.abs() > (1.0 - 1e-6)
        up = default_up.repeat(N, 1)
        if parallel_mask.any():
            up[parallel_mask] = alt_up  # only switch for the degenerate cases (e.g., poles)
    
    R, t = look_at_view_transform(eye=eyes, at=at, up=up, device=device)
    return R, t


def farthest_point_sampling(points: torch.Tensor, k: int, random_start: bool = True, start_idx: int = None) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) on a single point cloud.

    Args:
        points: (N, 3) float tensor of 3D points.
        k:     number of points to sample, 1 <= k <= N.
        random_start: if True, seed FPS with a random point; otherwise use the point
                      farthest from the centroid (slightly more deterministic).
        start_idx: if not None, start the sampling from the given index.

    Returns:
        (k,) long tensor of indices into `points` for the sampled points.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` must be of shape (N, 3); got {tuple(points.shape)}")
    N = points.shape[0]
    if not (1 <= k <= N):
        raise ValueError(f"`k` must satisfy 1 <= k <= N; got k={k}, N={N}")

    device = points.device
    pts = points.float()

    # Track the minimum squared distance of each point to the current sampled set
    min_dist2 = torch.full((N,), float("inf"), device=device)
    chosen = torch.empty(k, dtype=torch.long, device=device)

    # Pick the initial index
    if start_idx is None:
        if random_start:
            start_idx = torch.randint(N, (1,), device=device).item()
        else:
            centroid = pts.mean(dim=0, keepdim=True)         # (1, 3)
            dist2_centroid = ((pts - centroid) ** 2).sum(1)  # (N,)
            start_idx = torch.argmax(dist2_centroid).item()

    far_idx = start_idx
    for i in range(k):
        chosen[i] = far_idx
        # Update each point's distance to the closest sampled point so far
        dist2 = ((pts - pts[far_idx]) ** 2).sum(dim=1)   # (N,)
        min_dist2 = torch.minimum(min_dist2, dist2)
        # Next farthest point is the one maximizing its distance to the sampled set
        # Mask out already-chosen indices by setting their distance to -inf
        min_dist2[chosen[: i + 1]] = float("-inf")
        far_idx = torch.argmax(min_dist2).item()

    return chosen