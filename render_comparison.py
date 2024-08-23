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
from scene import Scene, DynamicScene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(render_path, iteration, views, gaussians, pipeline, background):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    if not os.path.exists(render_path):
        os.makedirs(render_path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        # gt = view.original_image[0:3, :, :]
        if view.image_name not in ['0', '18', '27', '36', '45', '54', '63', '72', '9']:
            continue
        save_name = view.image_name
        torchvision.utils.save_image(rendering, os.path.join(render_path, save_name + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    scene = DynamicScene(model.extract(args), shuffle=False)

    dataset = model.extract(args)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, scene)

    start = 10
    end = 19
    # qp_num = 30
    qp_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    name = "ykx_wo_entropy"

    for qp_num in qp_list:

        qp = f"qp_{qp_num}"
        ckpt_path = f'/home/auwang/workspace/VideoDynamicGaussian/ablation/{name}/recover'
        render_path = f'/home/auwang/workspace/VideoDynamicGaussian/ablation/{name}/render'
        for frame in range(start, end + 1):
            with torch.no_grad():
                gaussians = GaussianModel(0)
                # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                gaussians.load_ply(os.path.join(ckpt_path, str(qp), f"point_cloud_{frame}.ply"))

                bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                frame_render_path = os.path.join(render_path, str(qp), f"{frame}")
                render_set(frame_render_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)