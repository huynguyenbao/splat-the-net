import os
import uuid
import torch
from random import randint
import matplotlib.pyplot as plt
from utils.loss_utils import (
    l1_loss,
    ssim,
    density_loss,
    anisotropy_loss,
    fused_ssim,
    EdgeLoss,
)
from lpipsPyTorch import lpips
from gaussian_renderer import network_gui
import sys
from scene import Scene
from utils.general_utils import safe_state, density2alpha
from utils.sh_utils import C0
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import numpy as np
import random


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if args.render_backend in ["slang_volr"]:
        gaussians.init_opacity_volr = opt.init_opacity_volr
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    if checkpoint and os.path.exists(checkpoint):
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    gaussians.reparam_type = args.reparam_type

    gaussians.use_positional_grad = opt.use_positional_grad
    gaussians.not_use_mlp_grad = opt.not_use_mlp_grad

    gaussians.setup_functions()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    # Pick a random Camera
    ih, iw = (
        scene.getTrainCameras()[0].image_height,
        scene.getTrainCameras()[0].image_width,
    )
    for i in range(len(scene.getTrainCameras())):
        ihh, iww = (
            scene.getTrainCameras()[i].image_height,
            scene.getTrainCameras()[i].image_width,
        )
        if ihh != ih or iww != iw:
            print(
                f"Camera {i} has different image size: {ihh}x{iww} instead of {ih}x{iw}"
            )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = gaussian_renderer.render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 2000 its we increase the levels of SH up to a maximum degree
        if iteration % 2000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(num_cams=args.num_cams).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = gaussian_renderer.render(
            viewpoint_cam, gaussians, pipe, bg, opt.scale_modifier
        )
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            radii,
            frequencies_tensor,
            phases_tensor,
            amplitudes_tensor,
            offsets_tensor,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["frequencies"],
            render_pkg["phases"],
            render_pkg["amplitudes"],
            render_pkg["offsets"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )

        if opt.use_regularizer and iteration >= opt.regularizer_iter_start:
            # scale_reg_loss = torch.std(scene.gaussians.get_scaling, dim=1).mean()
            # loss += opt.lambda_anisotropy * scale_reg_loss

            if opt.lambda_anisotropy > 0:
                min_scaling = torch.min(scene.gaussians.get_scaling, 1).values
                # scale_reg_loss = torch.std(
                #     scene.gaussians.get_scaling / min_scaling[:, None], dim=1
                # ).mean()

                # Compute the scaling ratio
                scaling_ratio = scene.gaussians.get_scaling / min_scaling[:, None]

                # Create a mask for rows without NaNs
                valid_mask = ~torch.isnan(scaling_ratio).any(dim=1)

                # Apply the mask to filter valid rows
                valid_scaling_ratio = scaling_ratio[valid_mask]

                # Compute the regularization loss only on valid rows
                scale_reg_loss = torch.std(valid_scaling_ratio, dim=1).mean()

                loss += opt.lambda_anisotropy * scale_reg_loss

        loss.backward()
        if args.render_backend in ["slang_volr"]:
            viewspace_point_tensor = gaussians.get_xyz

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (
                iteration < opt.densify_until_iter
                and gaussians.get_xyz.shape[0] < args.max_points
            ):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    frequencies_tensor,
                    phases_tensor,
                    amplitudes_tensor,
                    offsets_tensor,
                    visibility_filter,
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    print("Start densification")
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_clone_grad_threshold,
                        opt.densify_split_grad_threshold,
                        opt.min_grad_prune,
                        scene.cameras_extent * args.scene_extent_mult,
                        size_threshold,
                    )

                if not opt.disable_opacity_reset and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                gaussian_renderer.render,
                (pipe, background),
            )

            # Optimizer step
            if iteration < opt.iterations:
                if iteration % opt.optimize_every == 0:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                    gaussians.remove_nan_points()

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
            if (
                args.checkpoint_interval > 0
                and iteration % args.checkpoint_interval == 0
            ):
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + "latest" + ".pth",
                )

            # return


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer and iteration % 200 == 0:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_histogram(
            "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
        )

        if hasattr(scene.gaussians, "_opacity_volr"):
            tb_writer.add_histogram(
                "scene/opacity_volr_histogram",
                scene.gaussians.get_opacity_volr,
                iteration,
            )
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )

                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

                if psnr_test > scene.best_psnr and config["name"] == "test":
                    scene.best_psnr = psnr_test
                    scene.best_iteration = iteration
                    scene.save("best")
                    print(
                        "\nSaving best psnr {} at iteration {}".format(
                            scene.best_psnr, scene.best_iteration
                        )
                    )
                print(
                    f"Best ckpt is now at {scene.best_iteration} iteration, {scene.best_psnr} psnr"
                )

        # if tb_writer:
        #     tb_writer.flush()

    if tb_writer:
        tb_writer.add_histogram(
            "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
        )
        if hasattr(scene.gaussians, "_opacity_volr"):
            tb_writer.add_histogram(
                "scene/opacity_volr_histogram",
                scene.gaussians.get_opacity_volr,
                iteration,
            )
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[
            7_000,
            10_000,
            20_000,
            30_000,
            40_000,
            50_000,
            60_000,
            70_000,
            80_000,
            90_000,
            100_000,
        ],
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
    parser.add_argument("--abs_gs", action="store_true", default=False)
    parser.add_argument("--num_cams", type=int, default=-1)
    parser.add_argument("--max_points", type=int, default=7e6)
    parser.add_argument("--scene_extent_mult", type=float, default=1.0)
    parser.add_argument(
        "--reparam_type", type=str, default="ours", choices=["ours", "ever", "None"]
    )
    parser.add_argument("--disable_random_suffix", action="store_false")
    parser.add_argument("--checkpoint_interval", type=int, default=-1)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.render_backend == "inria_cuda":
        from scene import GaussianModel
        import gaussian_renderer as gaussian_renderer
    elif args.render_backend == "slang":
        from scene import GaussianModel
        import slang_gaussian_rasterization.api.inria_3dgs as gaussian_renderer
    elif args.render_backend == "slang_volr":
        import slang_gaussian_rasterization.api.inria_3dgs_volr as gaussian_renderer
        from scene import GaussianModelVolr as GaussianModel
    elif args.render_backend == "inria_cuda_mlp":
        from scene import GaussianModelMLP_HB as GaussianModel
        import gaussian_mlp_max_renderer as gaussian_renderer

    # Generate a random UUID and take the first 6 characters
    if args.disable_random_suffix:
        random_suffix = str(uuid.uuid4())[:6]
        args.model_path = f"{args.model_path}_{random_suffix}"
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )

    # All done
    print("\nTraining complete.")
