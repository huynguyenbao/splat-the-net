#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import sys
from datetime import datetime
import numpy as np
import random

def alpha2density(alpha, scaling, reparam_type="ours"):
    """ Function to convert an alpha value 0-1 to a density (0-infinity) 
    Args:
        alpha (torch.Tensor [N]): Alpha value between 0 and 1
        scaling (torch.Tensor [N,3]): Scaling factor
    Returns:
        torch.Tensor: Density value
    """
    if reparam_type == "None":
        return alpha ## assume alpha is density 
    if reparam_type == "ever":
        density = (-torch.log(1.0 - 0.99*alpha))
        density = density / ((torch.min(scaling, dim=-1, keepdim=True).values))
        return density
    if reparam_type == "ours":  
        density = (-torch.log(1.0 - 0.99*alpha))
        density = density * ((torch.mean(1.0/scaling, dim=-1, keepdim=True))) / 2.506
        if torch.isnan(density).any():
            print("NaN detected in alpha")
        if (alpha < 0).any() or (alpha > 1).any():
            print("Alpha values outside valid range [0,1]")
        return density
    return density


def density2alpha(density, scaling, reparam_type="ours"):
    """ Function to convert an alpha value 0-1 to a density (0-infinity) 
    Args:
        alpha (torch.Tensor [N]): Alpha value between 0 and 1
        scaling (torch.Tensor [N,3]): Scaling factor
    Returns:
        torch.Tensor: Density value
    """
    if reparam_type == "None":
        return density
    if reparam_type == "ever":
        alpha = (1 - torch.exp(-density * torch.max(scaling, dim=-1, keepdim=True).values)).clamp(min=0.0, max=0.9999 )
    if reparam_type == "ours":
        alpha = (1 - torch.exp(-density * 2.506 * 1.0/((torch.mean(1.0/scaling, dim=-1, keepdim=True)))) / 0.99).clamp(min=0.0, max=0.99999)
        if torch.isnan(alpha).any():
            print("NaN detected in alpha")
        if (alpha < 0).any() or (alpha > 1).any():
            print("Alpha values outside valid range [0,1]")
    return alpha

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def compute_eccentricities(tensor):
    # Step 1: Sort each row to ensure a >= b >= c
    sorted_tensor, _ = torch.sort(tensor, descending=True, dim=1)
    a, b, c = sorted_tensor[:, 0], sorted_tensor[:, 1], sorted_tensor[:, 2]
    
    # Step 2: Compute the eccentricities
    ecc_ac = torch.sqrt(1 - (c ** 2) / (a ** 2))  # Between a and c
    ecc_ab = torch.sqrt(1 - (b ** 2) / (a ** 2))  # Between a and b
    
    return ecc_ac, ecc_ab

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    torch.use_deterministic_algorithms(True)


def create_points_in_sphere(N):
    # for n points in a sphere, we need 2n points in the cube.
    n = int((2 * N)**(1/3))
    # Create a 3D grid of points in a 2x2x2 cube
    eps = 0.01
    x = torch.linspace(-2 + eps, 2 - eps, n, dtype=torch.float, device="cuda")
    y = torch.linspace(-2 + eps, 2 - eps, n, dtype=torch.float, device="cuda") 
    z = torch.linspace(-2 + eps, 2 - eps, n, dtype=torch.float, device="cuda")
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    # Stack coordinates into points
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # Calculate distance from origin for each point
    distances = torch.norm(points, dim=1)

    # Keep only points inside sphere of radius 2
    mask = distances <= 2.0
    points = points[mask]

    closest_distance = 4 / N

    def C_inverse(z):
        """
        Computes C^(-1)(z) according to equation (10) for N points independently
        
        Args:
            z (torch.Tensor): Input tensor of shape (N, 3)
            
        Returns:
            torch.Tensor: Result of C^(-1)(z) with shape (N, 3)
        """
        # Compute L2 norm along dim=1 to get shape (N,)
        norm_z = torch.norm(z, dim=1)
        # breakpoint()
        
        # Compute terms for each point, broadcasting to shape (N,)
        max_term = torch.max(torch.ones_like(norm_z), norm_z**2)
        sqrt_max_term = torch.sqrt(max_term)
        clamped_sqrt_max_term = torch.clamp(sqrt_max_term, max=2.0)
        # breakpoint()
        
        # Compute denominator and reshape to (N,1) for broadcasting
        denominator = (sqrt_max_term * (2 - clamped_sqrt_max_term)).unsqueeze(-1)
        # breakpoint()
        
        # Divide each xyz coordinate by its corresponding denominator
        return z / denominator

    points_inv = C_inverse(points)
    scales = (0.3 / (2 - torch.norm(points, dim=1))).unsqueeze(-1).repeat(1, 3).log()

    return points_inv, scales