
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras
)

def simulate_scan_single_camera(V, F, N, R, t, K, H, W, device='cuda', cull_backfaces=False, flip_back_normals=False):
    """
    Surface point and normal extraction.
    """
    # Move tensors to device
    V = V.to(device).float()
    F = F.to(device)
    if N is not None:
        N = N.to(device).float()
    R = R.to(device).float()
    t = t.to(device).float()
    K = K.to(device).float()
    
    # Ensure proper dimensions
    if V.dim() == 2:
        V = V.unsqueeze(0)
    if F.dim() == 2:
        F = F.unsqueeze(0)
    if N is not None and N.dim() == 2:
        N = N.unsqueeze(0)
    
    # Extract camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    focal_length = torch.tensor([[fx, fy]], device=device)
    principal_point = torch.tensor([[cx, cy]], device=device)
    image_size = torch.tensor([[H, W]], device=device)
    
    # Create camera
    cameras = PerspectiveCameras(
        R=R.unsqueeze(0),
        T=t.unsqueeze(0),
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
        device=device
    )
    
    # Create mesh
    meshes = Meshes(verts=V, faces=F)
    
    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
        bin_size=0,
        cull_backfaces=cull_backfaces
    )
    
    # Rasterize
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)
    
    # Extract rasterization outputs
    pix_to_face = fragments.pix_to_face[0, :, :, 0]  # (H, W)
    bary_coords = fragments.bary_coords[0, :, :, 0, :]  # (H, W, 3)
    
    # Get mesh data
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    if N is None:
        normals_packed = meshes.verts_normals_packed()
    else:
        normals_packed = N[0]
    
    # Vectorized interpolation
    valid_mask = pix_to_face >= 0
    valid_faces = pix_to_face[valid_mask].long()
    
    # Get vertex indices
    face_verts = faces_packed[valid_faces]  # (N_valid, 3)
    
    # Get vertex positions and normals
    v0 = verts_packed[face_verts[:, 0]]
    v1 = verts_packed[face_verts[:, 1]]
    v2 = verts_packed[face_verts[:, 2]]
    
    n0 = normals_packed[face_verts[:, 0]]
    n1 = normals_packed[face_verts[:, 1]]
    n2 = normals_packed[face_verts[:, 2]]
    
    # Get barycentric coordinates for valid pixels
    valid_bary = bary_coords[valid_mask]  # (N_valid, 3)
    
    # Interpolate positions
    interpolated_points = (valid_bary[:, 0:1] * v0 + 
                          valid_bary[:, 1:2] * v1 + 
                          valid_bary[:, 2:3] * v2)
    
    # Interpolate normals
    interpolated_normals = (valid_bary[:, 0:1] * n0 + 
                           valid_bary[:, 1:2] * n1 + 
                           valid_bary[:, 2:3] * n2)
    interpolated_normals = torch.nn.functional.normalize(interpolated_normals, dim=1)

    # Optionally flip back-facing normals to point toward the camera
    if flip_back_normals and not cull_backfaces and valid_mask.any():
        # Camera center in world coordinates
        cam_center = cameras.get_camera_center()[0]  # (3,)
        # try:
        #     cam_center = cameras.get_camera_center()[0]  # (3,)
        # except:
        #     print(f"[WARNING] Failed to get camera center")
        #     print(R)
        #     print(t)
        #     exit()
        view_dirs = cam_center.unsqueeze(0) - interpolated_points  # (N_valid,3)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=1)
        dot = (interpolated_normals * view_dirs).sum(dim=1)
        flip_mask = dot < 0
        if flip_mask.any():
            interpolated_normals[flip_mask] = -interpolated_normals[flip_mask]

    # Filter out zero normals
    zero_normal_mask = interpolated_normals.norm(dim=1) < 1e-6
    if zero_normal_mask.any():
        print(f"[WARNING] {zero_normal_mask.sum()} zero normals found")
        valid_mask[valid_mask][zero_normal_mask] = False
    
    # Create output tensors
    surface_points = torch.zeros(H, W, 3, device=device)
    surface_normals = torch.zeros(H, W, 3, device=device)
    
    # Fill in valid pixels
    surface_points[valid_mask] = interpolated_points
    surface_normals[valid_mask] = interpolated_normals
    
    # Depth map
    depth_map = fragments.zbuf[0, :, :, 0]
    depth_map[depth_map < 0] = 0
    
    return surface_points, surface_normals, depth_map, valid_mask

def simulate_scan_batched(V, F, N, R, t, K, H, W, device='cuda', cull_backfaces=False, flip_back_normals=False):
    """
    Vectorized surface point and normal extraction supporting multiple cameras in one call.
    This version is differentiable with respect to camera parameters R and t.

    Inputs:
        V: (V,3) or (1,V,3) float tensor
        F: (F,3) or (1,F,3) long tensor
        N: (V,3) or (1,V,3) float tensor, or None
        R: (B,3,3) or (3,3) tensor
        t: (B,3) or (3,) tensor
        K: (B,3,3) or (3,3) tensor
        H, W: ints
    Returns:
        surface_points: (B,H,W,3)
        surface_normals: (B,H,W,3)
        depth_map: (B,H,W)
        mask: (B,H,W) float
    """
    # Move mesh tensors
    V = V.to(device).float()
    F = F.to(device)
    if N is not None:
        N = N.to(device).float()

    # Normalize mesh batch dims
    if V.dim() == 2:
        V = V.unsqueeze(0)
    if F.dim() == 2:
        F = F.unsqueeze(0)
    if N is not None and N.dim() == 2:
        N = N.unsqueeze(0)

    # Handle batched cameras
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)

    R = R.to(device).float()
    t = t.to(device).float()
    K = K.to(device).float()

    batch_size = R.shape[0]

    # Build intrinsics tensors
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    focal_length = torch.stack([fx, fy], dim=1)  # (B,2)
    principal_point = torch.stack([cx, cy], dim=1)  # (B,2)
    image_size = torch.full((batch_size, 2), fill_value=0.0, device=device)
    image_size[:, 0] = H
    image_size[:, 1] = W

    # Create cameras
    cameras = PerspectiveCameras(
        R=R,
        T=t,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
        device=device
    )

    # Create mesh and expand to batch if needed
    meshes = Meshes(verts=V, faces=F, verts_normals=N)
    if len(meshes) == 1 and batch_size > 1:
        meshes = meshes.extend(batch_size)

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
        bin_size=0,
        cull_backfaces=cull_backfaces
    )

    # Rasterize
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)

    # Initialize outputs with proper gradients - use zeros_like to maintain gradient flow
    surface_points = torch.zeros(batch_size, H, W, 3, device=device, requires_grad=R.requires_grad)
    surface_normals = torch.zeros(batch_size, H, W, 3, device=device, requires_grad=R.requires_grad)
    mask = torch.zeros(batch_size, H, W, dtype=torch.bool, device=device)
    depth_map = fragments.zbuf[:, :, :, 0]
    depth_map = torch.where(depth_map < 0, torch.zeros_like(depth_map), depth_map)

    # Mesh packed data
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    normals_packed = meshes.verts_normals_packed()

    # Vectorized interpolation for all batches
    pix_to_face = fragments.pix_to_face[:, :, :, 0]  # (B,H,W)
    bary_coords = fragments.bary_coords[:, :, :, 0, :]  # (B,H,W,3)
    valid_mask = pix_to_face >= 0  # (B,H,W)

    # Create coordinate grids for scatter operations
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, H, W)
    height_indices = torch.arange(H, device=device).view(1, H, 1).expand(batch_size, H, W)
    width_indices = torch.arange(W, device=device).view(1, 1, W).expand(batch_size, H, W)

    # Flatten for easier processing
    batch_flat = batch_indices.flatten()
    height_flat = height_indices.flatten()
    width_flat = width_indices.flatten()
    pix_to_face_flat = pix_to_face.flatten()
    bary_coords_flat = bary_coords.view(batch_size * H * W, 3)
    valid_mask_flat = valid_mask.flatten()

    # Get valid elements
    valid_indices = torch.where(valid_mask_flat)[0]
    if len(valid_indices) == 0:
        return surface_points, surface_normals, depth_map, mask

    valid_faces = pix_to_face_flat[valid_indices].long()
    valid_bary = bary_coords_flat[valid_indices]
    valid_batch = batch_flat[valid_indices]
    valid_height = height_flat[valid_indices]
    valid_width = width_flat[valid_indices]

    # Get vertex positions and normals for valid faces
    face_verts = faces_packed[valid_faces]
    v0 = verts_packed[face_verts[:, 0]]
    v1 = verts_packed[face_verts[:, 1]]
    v2 = verts_packed[face_verts[:, 2]]

    n0 = normals_packed[face_verts[:, 0]]
    n1 = normals_packed[face_verts[:, 1]]
    n2 = normals_packed[face_verts[:, 2]]

    # Interpolate positions and normals
    interp_points = (valid_bary[:, 0:1] * v0 +
                     valid_bary[:, 1:2] * v1 +
                     valid_bary[:, 2:3] * v2)
    interp_normals = (valid_bary[:, 0:1] * n0 +
                      valid_bary[:, 1:2] * n1 +
                      valid_bary[:, 2:3] * n2)
    interp_normals = torch.nn.functional.normalize(interp_normals, dim=1)

    # Optionally flip back-facing normals to point toward the camera
    if flip_back_normals and not cull_backfaces:
        # Get camera centers for all valid pixels
        cam_centers = cameras.get_camera_center()[valid_batch]  # (N_valid, 3)
        view_dirs = cam_centers - interp_points  # (N_valid, 3)
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=1)
        dot = (interp_normals * view_dirs).sum(dim=1)
        flip_mask = dot < 0
        # Use differentiable conditional operation
        flip_factor = torch.where(flip_mask, -1.0, 1.0).unsqueeze(1)
        interp_normals = interp_normals * flip_factor

    # Filter zero normals using differentiable operations
    normal_norms = interp_normals.norm(dim=1)
    zero_normal_mask = normal_norms < 1e-6
    if zero_normal_mask.any():
        print(f"[WARNING] {zero_normal_mask.sum()} zero normals found")
        # Create a mask that excludes zero normals
        keep_mask = ~zero_normal_mask
        valid_indices = valid_indices[keep_mask]
        interp_points = interp_points[keep_mask]
        interp_normals = interp_normals[keep_mask]
        valid_batch = valid_batch[keep_mask]
        valid_height = valid_height[keep_mask]
        valid_width = valid_width[keep_mask]

    # Use scatter operations to assign values (differentiable)
    if len(valid_indices) > 0:
        # Create linear indices for scatter operations
        linear_indices = valid_batch * H * W + valid_height * W + valid_width
        
        # Scatter surface points - use scatter_add for better gradient flow
        surface_points_flat = surface_points.view(batch_size * H * W, 3)
        surface_points_flat = surface_points_flat.scatter(0, linear_indices.unsqueeze(1).expand(-1, 3), interp_points)
        surface_points = surface_points_flat.view(batch_size, H, W, 3)
        
        # Scatter surface normals
        surface_normals_flat = surface_normals.view(batch_size * H * W, 3)
        surface_normals_flat = surface_normals_flat.scatter(0, linear_indices.unsqueeze(1).expand(-1, 3), interp_normals)
        surface_normals = surface_normals_flat.view(batch_size, H, W, 3)

    # Update mask
    mask = valid_mask

    return surface_points, surface_normals, depth_map, mask