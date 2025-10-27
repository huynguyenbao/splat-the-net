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
import math
import torch
import numpy as np
from utils.general_utils import (
    inverse_sigmoid,
    inverse_softplus,
    get_expon_lr_func,
    build_rotation,
    alpha2density,
    density2alpha,
    create_points_in_sphere,
)
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import matplotlib.pyplot as plt


class GaussianModelMLP_HB:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.opacity_volr_activation = (
            torch.sigmoid
            if self.reparam_type == "ours" or self.reparam_type == "ever"
            else torch.nn.functional.softplus
        )
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_opacity_volr_activation = (
            inverse_sigmoid
            if self.reparam_type == "ours" or self.reparam_type == "ever"
            else inverse_softplus
        )
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, n_neurons: int = 8):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._opacity_volr = torch.empty(0)
        self._abs_xyz_var = torch.empty(0)

        # + ---------------------- +
        # |       New Code         |
        # + ---------------------- +
        self._frequencies = torch.empty(0)  # [points, n_neurons, 3]
        self._phases = torch.empty(0)  # [points, n_neurons]
        self._amplitudes = torch.empty(0)  # [points, n_neurons]
        self._offsets = torch.empty(0)  # [points]
        self.n_neurons = n_neurons
        self.omega = 30
        self.use_positional_grad = False
        self.not_use_mlp_grad = False
        # + ---------------------- +
        # |       New Code         |
        # + ---------------------- +

        self.init_opacity_volr = 1000.0
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.mlp_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.reparam_type = "ours"
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_volr,
            self._frequencies,
            self._phases,
            self._amplitudes,
            self._offsets,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.mlp_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_volr,
            self._frequencies,
            self._phases,
            self._amplitudes,
            self._offsets,
            self.max_radii2D,
            xyz_gradient_accum,
            mlp_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.mlp_gradient_accum = mlp_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_frequencies(self):
        return self._frequencies

    @property
    def get_phases(self):
        return self._phases

    @property
    def get_amplitudes(self):
        return self._amplitudes

    @property
    def get_offsets(self):
        return self._offsets

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_volr(self):
        density = alpha2density(
            self.opacity_volr_activation(self._opacity_volr),
            self.get_scaling,
            self.reparam_type,
        )
        return density

    @property
    def get_density(self):
        density = alpha2density(
            self.opacity_volr_activation(self._opacity_volr),
            self.get_scaling,
            self.reparam_type,
        )
        return density

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
        self, pcd: BasicPointCloud, spatial_lr_scale: float, toy_example: bool = False
    ):
        # self.spatial_lr_scale = spatial_lr_scale
        self.spatial_lr_scale = 1

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # fused_color = RGB2SH(torch.ones_like(torch.tensor(np.asarray(pcd.colors)).float().cuda()) * torch.tensor([[0.2, 0.3, 0.5]]).float().cuda())
        # fused_color = RGB2SH(torch.ones_like(torch.tensor(np.asarray(pcd.colors)).float().cuda()))

        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # + ---------------------- +
        # |       New Code         |
        # + ---------------------- +
        std_freq = 1.0 / 3
        std_ampl = math.sqrt(6 / self.n_neurons) / self.omega
        frequencies = (
            torch.rand((fused_color.shape[0], 3, self.n_neurons)).float().cuda()
            * 2
            * std_freq
            - std_freq
        )
        phases = (
            torch.rand((fused_color.shape[0], self.n_neurons)).float().cuda()
            * 2
            * std_freq
            - std_freq
        )
        amplitudes = (
            torch.rand((fused_color.shape[0], self.n_neurons)).float().cuda()
            * 2
            * std_ampl
            - std_ampl
        )
        offsets = (
            torch.rand((fused_color.shape[0], 1)).float().cuda() * 2 * std_ampl
            - std_ampl
        )

        # + ---------------------- +
        # |       New Code         |
        # + ---------------------- +

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        if toy_example:
            scales = torch.log(torch.sqrt(dist2) / 20)[..., None].repeat(1, 3)
            offsets += 1

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities_volr = self.inverse_opacity_volr_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rots.contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self._opacity_volr = nn.Parameter(
            opacities_volr.contiguous().requires_grad_(True)
        )

        self._frequencies = nn.Parameter(
            frequencies.transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._phases = nn.Parameter(phases.contiguous().requires_grad_(True))
        self._amplitudes = nn.Parameter(amplitudes.contiguous().requires_grad_(True))
        self._offsets = nn.Parameter(offsets.contiguous().requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.mlp_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._opacity_volr],
                "lr": training_args.opacity_volr_lr,
                "name": "opacity_volr",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._frequencies],
                "lr": training_args.frequencies_mlp_lr,
                "name": "frequencies",
            },
            {
                "params": [self._phases],
                "lr": training_args.phases_mlp_lr,
                "name": "phases",
            },
            {
                "params": [self._amplitudes],
                "lr": training_args.amplitudes_mlp_lr,
                "name": "amplitudes",
            },
            {
                "params": [self._offsets],
                "lr": training_args.amplitudes_mlp_lr,
                "name": "offsets",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def clip_gradients(self):
        """Clip gradients to handle NaN values"""
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                param = param_group["params"][0]
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad)
                    if nan_mask.any():
                        param.grad[nan_mask] = 0.0

    def count_nan_gradients(self):
        """Count number of NaNs in gradients"""
        num_nans = 0
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                param = param_group["params"][0]
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad)
                    num_nans += torch.sum(nan_mask).item()
        return num_nans

    def any_nan(self):
        """check if anything is NaN"""
        try:
            with torch.no_grad():
                # Move tensors to CPU before checking to avoid CUDA memory issues
                for param_group in self.optimizer.param_groups:
                    param = param_group["params"][0]
                    # Check parameter values for NaNs
                    if torch.isnan(param).any():
                        return True
                        # Check parameter gradients for NaNs if they exist
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            return True
                    else:
                        if torch.isnan(param).any():
                            return True
                        if param.grad is not None and torch.isnan(param.grad).any():
                            return True
            return False
        except Exception as e:
            print(f"Warning: Error checking for NaNs: {e}")
            # If we can't check for NaNs, assume there aren't any
            return False

    def count_nans(self):
        num_nans = 0
        with torch.no_grad():
            for param_group in self.optimizer.param_groups:
                param = param_group["params"][0]
                nan_mask = torch.isnan(param.grad) | torch.isnan(param)
                num_nans += torch.sum(nan_mask).item()
        return num_nans

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        l.append("opacity_volr")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))

        # ----
        for i in range(self._frequencies.shape[1] * self._frequencies.shape[2]):
            l.append("frequencies_{}".format(i))
        for i in range(self._phases.shape[1]):
            l.append("phases_{}".format(i))
        for i in range(self._amplitudes.shape[1]):
            l.append("amplitudes_{}".format(i))
        l.append("offsets")
        # ----

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = self._opacity.detach().cpu().numpy()
        opacities_volr = self._opacity_volr.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # ---
        frequencies = (
            self._frequencies.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        phases = self._phases.detach().cpu().numpy()
        amplitudes = self._amplitudes.detach().cpu().numpy()
        offsets = self._offsets.detach().cpu().numpy()
        # ---

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                opacities_volr,
                scale,
                rotation,
                frequencies,
                phases,
                amplitudes,
                offsets,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # self._opacity = optimizable_tensors["opacity"]
        # opacities_volr_new = self.inverse_opacity_volr_activation(torch.min(0.01*torch.ones_like(self.get_opacity_volr),
        #                                                                     self.get_opacity_volr/self.max_opacity_volr))
        # breakpoint()
        opacities_volr_new = self.inverse_opacity_volr_activation(
            torch.min(
                0.01 * torch.ones_like(self.get_opacity_volr),
                self.opacity_volr_activation(self._opacity_volr),
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_volr_new, "opacity_volr"
        )
        self._opacity_volr = optimizable_tensors["opacity_volr"]

    def load_ply(
        self,
        path,
        skip_mlp=False,
        clamp_scale=False,
        min_scale=-5,
        removing_percent=-1,
        jitter=False,
    ):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities_volr = np.asarray(plydata.elements[0]["opacity_volr"])[
            ..., np.newaxis
        ]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        if clamp_scale:
            scales = np.clip(scales, min_scale, None)
        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # + ---------------------- +
        # |    MLP Data Loader     |
        # + ---------------------- +
        if skip_mlp:
            # Initialize mlp weights
            std_freq = 1.0 / 3
            std_ampl = math.sqrt(6 / self.n_neurons) / self.omega
            frequencies = (
                torch.rand((xyz.shape[0], 3, self.n_neurons)).float().cuda()
                * 2
                * std_freq
                - std_freq
            )
            phases = (
                torch.rand((xyz.shape[0], self.n_neurons)).float().cuda() * 2 * std_freq
                - std_freq
            )
            amplitudes = (
                torch.rand((xyz.shape[0], self.n_neurons)).float().cuda() * 2 * std_ampl
                - std_ampl
            )
            offsets = (
                torch.rand((xyz.shape[0], 1)).float().cuda() * 2 * std_ampl - std_ampl
            )
            self.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
        else:
            # Load mlp weights
            frequencies_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("frequencies_")
            ]
            frequencies_names = sorted(
                frequencies_names, key=lambda x: int(x.split("_")[-1])
            )
            assert len(frequencies_names) == 3 * self.n_neurons
            frequencies = np.zeros((xyz.shape[0], len(frequencies_names)))
            for idx, attr_name in enumerate(frequencies_names):
                frequencies[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            frequencies = frequencies.reshape((frequencies.shape[0], 3, self.n_neurons))

            phases_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("phases_")
            ]
            phases_names = sorted(phases_names, key=lambda x: int(x.split("_")[-1]))
            phases = np.zeros((xyz.shape[0], len(phases_names)))
            for idx, attr_name in enumerate(phases_names):
                phases[:, idx] = np.asarray(plydata.elements[0][attr_name])

            amplitudes_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("amplitudes_")
            ]
            amplitudes_names = sorted(
                amplitudes_names, key=lambda x: int(x.split("_")[-1])
            )
            amplitudes = np.zeros((xyz.shape[0], len(amplitudes_names)))
            for idx, attr_name in enumerate(amplitudes_names):
                amplitudes[:, idx] = np.asarray(plydata.elements[0][attr_name])

            offsets = np.asarray(plydata.elements[0]["offsets"])[..., np.newaxis]
            self.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")

        # + ---------------------- +
        # |         End            |
        # + ---------------------- +

        self._xyz = nn.Parameter(
            torch.tensor(
                xyz,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(
                features_dc,
                dtype=torch.float,
                device="cuda",
            )
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(
                features_extra,
                dtype=torch.float,
                device="cuda",
            )
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._opacity_volr = nn.Parameter(
            torch.tensor(
                opacities_volr,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                scales,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                rots,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )

        # + ---------------------- +
        # |    MLP Data Loader     |
        # + ---------------------- +
        self._frequencies = nn.Parameter(
            torch.tensor(
                frequencies,
                dtype=torch.float,
                device="cuda",
            )
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._phases = nn.Parameter(
            torch.tensor(
                phases,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._amplitudes = nn.Parameter(
            torch.tensor(
                amplitudes,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        self._offsets = nn.Parameter(
            torch.tensor(
                offsets,
                dtype=torch.float,
                device="cuda",
            ).requires_grad_(True)
        )
        # + ---------------------- +
        # |         End            |
        # + ---------------------- +

        # + ---------------------- +
        # |    Radomly delete      |
        # + ---------------------- +
        with torch.no_grad():
            if removing_percent > 0 and removing_percent < 1:
                num_to_select = int((1 - removing_percent) * xyz.shape[0])

                # Generate random indices
                indices = torch.randperm(xyz.shape[0], device="cuda")[:num_to_select]

                # Select the elements
                self._xyz.data = self._xyz.data[indices]
                self._features_dc.data = self._features_dc.data[indices]
                self._features_rest.data = self._features_rest.data[indices]
                self._opacity.data = self._opacity.data[indices]
                self._opacity_volr.data = self._opacity_volr.data[indices]
                self._rotation.data = self._rotation.data[indices]
                self._scaling.data = self._scaling.data[indices]
                self._frequencies.data = self._frequencies.data[indices]
                self._phases.data = self._phases.data[indices]
                self._amplitudes.data = self._amplitudes.data[indices]
                self._offsets.data = self._offsets.data[indices]
                self.max_radii2D = torch.zeros((self._xyz.data.shape[0]), device="cuda")

        with torch.no_grad():
            if jitter:

                def quaternion_to_rotation_matrix(q):
                    # q: [N, 4] with (w, x, y, z)
                    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

                    N = q.size(0)
                    R = torch.zeros(N, 3, 3, device=q.device)

                    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
                    R[:, 0, 1] = 2 * (x * y - z * w)
                    R[:, 0, 2] = 2 * (x * z + y * w)

                    R[:, 1, 0] = 2 * (x * y + z * w)
                    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
                    R[:, 1, 2] = 2 * (y * z - x * w)

                    R[:, 2, 0] = 2 * (x * z - y * w)
                    R[:, 2, 1] = 2 * (y * z + x * w)
                    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

                    return R

                R = quaternion_to_rotation_matrix(self.get_rotation)
                xyz_noise = (
                    torch.randn((R.shape[0], 3), device="cuda") * self.get_scaling
                )
                xyz_noise = torch.sum(R * xyz_noise[:, None, :], dim=-1)
                self._xyz.data += xyz_noise
        # + ---------------------- +
        # |         End            |
        # + ---------------------- +

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._opacity_volr = optimizable_tensors["opacity_volr"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._frequencies = optimizable_tensors["frequencies"]
        self._phases = optimizable_tensors["phases"]
        self._amplitudes = optimizable_tensors["amplitudes"]
        self._offsets = optimizable_tensors["offsets"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.mlp_gradient_accum = self.mlp_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_opacities_volr,
        new_scaling,
        new_rotation,
        new_frequencies,
        new_phases,
        new_amplitudes,
        new_offsets,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "opacity_volr": new_opacities_volr,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "frequencies": new_frequencies,
            "phases": new_phases,
            "amplitudes": new_amplitudes,
            "offsets": new_offsets,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._opacity_volr = optimizable_tensors["opacity_volr"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # ---
        self._frequencies = optimizable_tensors["frequencies"]
        self._phases = optimizable_tensors["phases"]
        self._amplitudes = optimizable_tensors["amplitudes"]
        self._offsets = optimizable_tensors["offsets"]
        # ---

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.mlp_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, info=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        # if self.reparam_type == "None":
        #     selected_opacity_mask = torch.zeros_like(
        #         self.get_opacity_volr.squeeze(-1), dtype=torch.bool
        #     )
        # else:
        #     selected_opacity_mask = (
        #         density2alpha(
        #             self.get_opacity_volr, self.get_scaling, self.reparam_type
        #         ).squeeze(-1)
        #         >= 0.99
        #     )

        # selected_pts_mask = torch.logical_or(
        #     selected_pts_mask, selected_opacity_mask.squeeze(-1)
        # )

        selected_opacity_mask = torch.zeros_like(
            self.get_opacity_volr.squeeze(-1), dtype=torch.bool
        )
        selected_pts_mask = torch.logical_or(
            selected_pts_mask, selected_opacity_mask.squeeze(-1)
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        alpha = density2alpha(
            self.get_opacity_volr[selected_pts_mask].repeat(N, 1) / 2,
            self.get_scaling[selected_pts_mask].repeat(N, 1),
            self.reparam_type,
        )
        new_opacity_volr = self.inverse_opacity_volr_activation(alpha)

        # ---
        new_frequencies = self._frequencies[selected_pts_mask].repeat(N, 1, 1)
        new_phases = self._phases[selected_pts_mask].repeat(N, 1)
        new_amplitudes = self._amplitudes[selected_pts_mask].repeat(N, 1)
        new_offsets = self._offsets[selected_pts_mask].repeat(N, 1)
        # ---

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_opacity_volr,
            new_scaling,
            new_rotation,
            new_frequencies,
            new_phases,
            new_amplitudes,
            new_offsets,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

        if info is not None:
            new_info = {}
            for name, tensor in info.items():
                tensor_selct = tensor[selected_pts_mask].repeat(N, 1)
                tensor_new = torch.concat([tensor, tensor_selct], dim=0)
                tensor_filtered = tensor_new[~prune_filter]
                new_info[name] = tensor_filtered

            return new_info

    def densify_and_clone(self, grads, grad_threshold, scene_extent, info=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,  #
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_opacities_volr = self.inverse_opacity_volr_activation(
            density2alpha(
                alpha2density(
                    self.opacity_volr_activation(self._opacity_volr[selected_pts_mask]),
                    self.get_scaling[selected_pts_mask],
                    self.reparam_type,
                )
                / 2.0,
                self.get_scaling[selected_pts_mask],
                self.reparam_type,
            )
        )

        new_scaling = self._scaling[
            selected_pts_mask
        ]  # / self.get_opacity_volr[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # ---
        new_frequencies = self._frequencies[selected_pts_mask]
        new_phases = self._phases[selected_pts_mask]
        new_amplitudes = self._amplitudes[selected_pts_mask]
        new_offsets = self._offsets[selected_pts_mask]
        # ---

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_opacities_volr,
            new_scaling,
            new_rotation,
            new_frequencies,
            new_phases,
            new_amplitudes,
            new_offsets,
        )

        if info is not None:
            new_info = {}
            for name, tensor in info.items():
                tensor_selct = tensor[selected_pts_mask]
                tensor_cloned = torch.concat([tensor, tensor_selct], dim=0)
                new_info[name] = tensor_cloned
            return new_info

    def densify_and_prune(
        self, max_grad_clone, max_grad_split, min_grad_prune, extent, max_screen_size
    ):
        # grads = self.xyz_gradient_accum / self.denom
        if self.not_use_mlp_grad:
            grads = 0
        else:
            grads = self.mlp_gradient_accum
        if self.use_positional_grad:
            grads = (grads + self.xyz_gradient_accum) / 2

        grads = grads / self.denom

        grads[grads.isnan()] = 0.0

        info = {
            "grads": grads.clone().detach(),
            "denom": self.denom.clone().detach(),
        }

        info_cloned = self.densify_and_clone(
            grads, max_grad_clone, extent, info=info
        )  # clone small gaussians

        info_filtered = self.densify_and_split(
            grads, max_grad_split, extent, N=2, info=info_cloned
        )  # split large gaussians

        # ---
        # opacity_volr_alpha = self.opacity_volr_activation(self._opacity_volr)
        # prune_mask = (opacity_volr_alpha < min_opacity).squeeze()

        # TODO: Update grads based on cloning and splitting.
        # -- FIX ME --
        # should consider visible primitives only (denom > 0)
        prune_mask = torch.bitwise_and(
            info_filtered["grads"] < min_grad_prune, info_filtered["denom"] > 0
        ).squeeze()
        # --

        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     # big_points_ws = self.get_scaling.max(dim=1).values  > 0.1 * extent
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.01 * extent
        #     prune_mask = torch.logical_or(
        #         torch.logical_or(prune_mask, big_points_vs), big_points_ws
        #     )

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        print("points: ", self.get_xyz.shape[0])

    def add_densification_stats(
        self,
        xyz_tensor,
        frequencies_tensor,
        phases_tensor,
        amplitudes_tensor,
        offsets_tensor,
        update_filter,
    ):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            xyz_tensor.grad[update_filter], dim=-1, keepdim=True
        )
        self.mlp_gradient_accum[update_filter] += torch.norm(
            frequencies_tensor.grad[update_filter], dim=(-2, -1)
        )[:, None] + torch.norm(
            amplitudes_tensor.grad[update_filter], dim=-1, keepdim=True
        )
        # (N, 1)

        self.mlp_gradient_accum[update_filter] += torch.norm(
            phases_tensor.grad[update_filter], dim=-1, keepdim=True
        ) + torch.norm(offsets_tensor.grad[update_filter], dim=-1, keepdim=True)

        self.denom[update_filter] += 1

    def save_opacity_scaling_histogram(self, iteration, output_dir):
        # Calculate the values
        values = (
            self.get_opacity_volr.squeeze(-1)
            * torch.max(self.get_scaling, dim=1).values
        )

        # Convert to numpy array
        values_np = values.detach().cpu().numpy()

        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(values_np, bins=1000, edgecolor="black")
        plt.title(f"Histogram of opacity_volr * max(scaling) at iteration {iteration}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.xscale("log")
        plt.yscale("log")
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        plt.savefig(
            os.path.join(output_dir, f"opacity_scaling_histogram_iter_{iteration}.png")
        )
        plt.close()

    def compute_densification_stats(
        self, max_grad, min_opacity, extent, max_screen_size
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        num_split_grad = selected_pts_mask.sum()
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent,
        )
        selected_opacity_mask = (
            density2alpha(
                self.get_opacity_volr, self.get_scaling, self.reparam_type
            ).squeeze(-1)
            >= 0.99
        )
        num_opaque_mask = selected_opacity_mask.sum()
        selected_opacity_mask = torch.logical_and(
            selected_opacity_mask,
            self.get_scaling.max(dim=1).values > self.percent_dense * extent,
        )
        selected_pts_mask = torch.logical_or(
            selected_pts_mask, selected_opacity_mask.squeeze(-1)
        )
        num_split_scale = (
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent
        ).sum()
        num_selected_pts = selected_pts_mask.sum()

        opacity_volr_alpha = self.opacity_volr_activation(self._opacity_volr)
        prune_mask = (opacity_volr_alpha < min_opacity).squeeze()
        num_prune_opacity = prune_mask.sum()

        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(
            torch.logical_or(prune_mask, big_points_vs), big_points_ws
        )

        num_prune_vs = big_points_vs.sum()
        num_prune_ws = big_points_ws.sum()
        return (
            num_split_grad,
            num_split_scale,
            num_selected_pts,
            num_prune_opacity,
            num_prune_vs,
            num_prune_ws,
            num_opaque_mask,
        )

    def get_split_mask(self, grad_threshold, scene_extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        return selected_pts_mask

    def get_clone_mask(self, grad_threshold, scene_extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        return selected_pts_mask

    def get_opacity_split_mask(self):
        selected_opacity_mask = (
            density2alpha(
                self.get_opacity_volr, self.get_scaling, self.reparam_type
            ).squeeze(-1)
            >= 0.99
        )
        return selected_opacity_mask

    def remove_nan_points(self):
        """Remove points that have NaN values in any parameter tensor"""
        # Create a mask for points with NaN values in any parameter
        nan_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")

        # Check each parameter tensor for NaN values
        params_to_check = [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_volr,
            self._frequencies,
            self._phases,
            self._amplitudes,
            self._offsets,
        ]

        for param in params_to_check:
            # Flatten all dimensions except the first (point dimension)
            param_flat = param.view(param.shape[0], -1)
            # Check for NaN in any dimension of each point
            param_nan_mask = torch.isnan(param_flat).any(dim=1)
            nan_mask = torch.logical_or(nan_mask, param_nan_mask)

        # Prune points with NaN values
        if nan_mask.sum() > 0:
            self.prune_points(nan_mask)
            print(f"Removed {nan_mask.sum().item()} points with NaN values")
