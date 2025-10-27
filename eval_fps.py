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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs

# from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import json


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    fps = 0
    n_repeats = 100
    warmup = 20

    # # Warmup phase (not timed)
    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     for _ in range(warmup):
    #         rendering = gaussian_renderer.render(view, gaussians, pipeline, background)["render"]

    torch.cuda.synchronize()
    start = time.time()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for _ in range(n_repeats):
            rendering = gaussian_renderer.render(view, gaussians, pipeline, background)[
                "render"
            ]

    torch.cuda.synchronize()
    end = time.time()

    elapsed_time = end - start
    total_frames = len(views) * n_repeats
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0
    n_primitives = gaussians.get_xyz.shape[0]

    all_data = {
        "fps": fps,
        "primitives": n_primitives,
    }

    json_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "summary_fps.json"
    )
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"fps: {fps}, primitives: {n_primitives}")
    print(f"Saving fps report at: {json_path}")


def render_sets(
    dataset: ModelParams,
    iteration: str,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    visualize_primitives: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        if visualize_primitives:
            gaussians.active_sh_degree = 0
            from utils.sh_utils import RGB2SH

            gaussians._features_dc.data = RGB2SH(
                torch.rand_like(gaussians._features_dc.data)
            )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--visualize_primitives", action="store_true")

    parser.add_argument(
        "--render_backend",
        type=str,
        default="inria_cuda_mlp",
        choices=[
            "slang",
            "slang_volr",
            "inria_cuda",
            "inria_cuda_mlp",
        ],
    )

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if args.render_backend == "inria_cuda":
        from scene import GaussianModel
        import gaussian_renderer as gaussian_renderer
    elif args.render_backend == "slang":
        from scene import GaussianModel
        import slang_gaussian_rasterization.api.inria_3dgs as gaussian_renderer
    elif args.render_backend == "slang_volr":
        from scene import GaussianModelVolr as GaussianModel
        import slang_gaussian_rasterization.api.inria_3dgs_volr as gaussian_renderer
    elif args.render_backend == "inria_cuda_mlp":
        from scene import GaussianModelMLP_HB as GaussianModel
        import gaussian_mlp_max_renderer as gaussian_renderer

    # Initialize system state (RNG)
    safe_state(args.quiet)
    pipeline_args = pipeline.extract(args)
    if args.render_backend in ["slang_volr"]:
        pipeline_args.softplus_rgb = True
        print("Using softplus RGB activation rendering slang_volr")
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline_args,
        args.skip_train,
        args.skip_test,
        visualize_primitives=args.visualize_primitives,
    )
