import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points
import point_cloud_utils as pcu
import torch.nn.functional as F

NUM_CLASSES = 16
NUM_PARTS = 50

SEG_CLASSES = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

CAT_LABEL = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

# SEG_LABEL_TO_CAT = {}  # {0:Airplane, 1:Airplane, ...49:Table}
# for cat in SEG_CLASSES.keys():
#     for label in SEG_CLASSES[cat]:
#         SEG_LABEL_TO_CAT[label] = cat

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_torch(pc):
    """
    pc: (N, 3) or (B, N, 3)
    """
    centroid = torch.mean(pc, dim=-2, keepdim=True) # (B, 1, 3)
    pc = pc - centroid # (B, N, 3)
    norm = torch.norm(pc, dim=-1, keepdim=True) # (B, N, 1)
    m, _ = torch.max(norm, dim=-2, keepdim=True) # (B, 1, 1)
    pc = pc / m
    return pc # (B, N, 3)

def subsample_points(points, normals, num_samples, fps=True, dup_pts=True, random_start_point=False):
    if num_samples is None:
        return points, normals
    num_points = points.shape[0]
    if num_points < num_samples:
        if dup_pts:
            dup_pts = points[0].detach().unsqueeze(0).expand(num_samples - num_points, 3)
            dup_nrms = normals[0].detach().unsqueeze(0).expand(num_samples - num_points, 3)
            points = torch.cat([points, dup_pts], dim=0)
            normals = torch.cat([normals, dup_nrms], dim=0)
        else:
            return points, normals
    else:
        if not fps:
            idx = torch.randperm(num_points)[:num_samples]
            points = points[idx]
            normals = normals[idx]
        else:
            points, idx = sample_farthest_points(points.unsqueeze(0), K=num_samples, random_start_point=random_start_point)
            points = points.squeeze(0)
            idx = idx.squeeze(0)
            normals = normals[idx]
    return points, normals

def model_predict(model, points, normals, detach=True):
    points = pc_normalize_torch(points)
    model_input = torch.cat([points, normals], dim=1) # (N, 6)
    model_input = model_input.unsqueeze(0).transpose(2, 1) # (1, 6, N)
    pred, _ = model(model_input)
    pred = pred.squeeze(0)
    pred_choice = pred.argmax().item()
    if detach:
        pred = pred.detach()
    return pred_choice, pred


def model_predict_batch(model, points, normals, detach=True):
    """
    points, normals: (B, N, 3)
    """
    # print(points.shape, normals.shape)
    points = pc_normalize_torch(points) # (B, N, 3)
    # print(points.shape)
    model_input = torch.cat([points, normals], dim=-1) # (B, N, 6)
    # print(model_input.shape)
    model_input = model_input.transpose(1, 2) # (B, 6, N)
    pred, _ = model(model_input) # (B, N_class)
    pred_choices = pred.argmax(dim=1) # (B,)
    if detach:
        pred = pred.detach()
    return pred_choices, pred


def model_predict_partseg(model, points, normals, cat, detach=True):
    """
    points, normals: (N, 3)
    cls_label: str
    """
    points = pc_normalize_torch(points)
    model_input = torch.cat([points, normals], dim=1) # (N, 6)
    model_input = model_input.unsqueeze(0).transpose(2, 1) # (1, 6, N)
    cls_label = CAT_LABEL[cat]
    cls_label = torch.tensor([cls_label], device=points.device) # (1,)
    cls_label_one_hot = F.one_hot(cls_label, num_classes=NUM_CLASSES).float().unsqueeze(1) # (1, 1, NUM_CLASSES)
    pred, _ = model(model_input, cls_label_one_hot, softmax=False) # (1, N, NUM_PARTS)
    pred = pred.squeeze(0) # (N, NUM_PARTS)
    if detach:
        pred = pred.detach()
    logits = pred[:, SEG_CLASSES[cat]] # (N, NUM_PARTS_IN_CAT)
    # logits = F.log_softmax(logits, dim=1) # (N, NUM_PARTS_IN_CAT)
    logits = F.softmax(logits, dim=1) # (N, NUM_PARTS_IN_CAT)
    return logits

def model_predict_partseg_batch(model, points, normals, cat, detach=True):
    """
    points, normals: (B, N, 3)
    cat: str
    """
    points = pc_normalize_torch(points) # (B, N, 3)
    model_input = torch.cat([points, normals], dim=-1) # (B, N, 6)
    model_input = model_input.transpose(1, 2) # (B, 6, N)
    cls_label = CAT_LABEL[cat]
    cls_label = torch.tensor([cls_label], device=points.device) # (1,)
    cls_label_one_hot = F.one_hot(cls_label, num_classes=NUM_CLASSES).float().unsqueeze(1) # (1, 1, NUM_CLASSES)
    cls_label_one_hot = cls_label_one_hot.expand(model_input.shape[0], -1, -1) # (B, 1, NUM_CLASSES)
    pred, _ = model(model_input, cls_label_one_hot, softmax=False) # (B, N, NUM_PARTS)
    if detach:
        pred = pred.detach()
    logits = pred[..., SEG_CLASSES[cat]] # (B, N, NUM_PARTS_IN_CAT)
    logits = F.softmax(logits, dim=-1) # (B, N, NUM_PARTS_IN_CAT)
    return logits


R_x = np.array([
    [1,  0, 0],
    [0,  0, 1],
    [0, -1, 0]
])

def align_mesh(V, F):
    V = V @ R_x.T
    V[:, 2] = -V[:, 2]
    F = F[:, [0, 2, 1]]
    return V, F

R_y = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])

def align_mesh_shapenet(V, F):
    V = V @ R_y.T
    V, F = pcu.make_mesh_watertight(V, F, 20000)
    return V, F