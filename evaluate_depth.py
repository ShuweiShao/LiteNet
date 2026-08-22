from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from options import MonodepthOptions
from utils import readlines


cv2.setNumThreads(0)

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4

TEST_FILES_PATH = "./test_files.txt"
GT_DEPTHS_PATH = "./gt_depths.npz"


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(left_disp, right_disp):
    _, height, width = left_disp.shape
    mean_disp = 0.5 * (left_disp + right_disp)

    left_grid, _ = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
    )
    left_mask = (
        1.0 - np.clip(20 * (left_grid - 0.05), 0, 1)
    )[None, ...]
    right_mask = left_mask[:, :, ::-1]

    return (
        right_mask * left_disp
        + left_mask * right_disp
        + (1.0 - left_mask - right_mask) * mean_disp
    )


def load_model_predictions(opt, device):
    load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(
            "Cannot find a folder at {}".format(load_weights_folder)
        )

    print("-> Loading weights from {}".format(load_weights_folder))

    filenames = readlines(TEST_FILES_PATH)
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)

    dataset = datasets.SCAREDRAWDataset(
        opt.data_path,
        filenames,
        encoder_dict["height"],
        encoder_dict["width"],
        [0],
        4,
        is_train=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    encoder = networks.LiteMono(
        model=opt.model,
        height=encoder_dict["height"],
        width=encoder_dict["width"],
    )
    depth_decoder = networks.DepthDecoder2(
        encoder.num_ch_enc,
        scales=range(3),
    )

    encoder_model_dict = encoder.state_dict()
    compatible_encoder_dict = {
        key: value
        for key, value in encoder_dict.items()
        if key in encoder_model_dict
    }
    encoder.load_state_dict(compatible_encoder_dict)
    depth_decoder.load_state_dict(
        torch.load(decoder_path, map_location=device)
    )

    encoder.to(device).eval()
    depth_decoder.to(device).eval()

    pred_disps = []

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].to(
                device,
                non_blocking=device.type == "cuda",
            )

            if opt.post_process:
                input_color = torch.cat(
                    (input_color, torch.flip(input_color, dims=[3])),
                    dim=0,
                )

            output = depth_decoder(encoder(input_color))
            pred_disp, _ = disp_to_depth(
                output[("disp", 0)],
                opt.min_depth,
                opt.max_depth,
            )
            pred_disp = pred_disp[:, 0].cpu().numpy()

            if opt.post_process:
                batch_size = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(
                    pred_disp[:batch_size],
                    pred_disp[batch_size:, :, ::-1],
                )

            pred_disps.append(pred_disp)

    return np.concatenate(pred_disps), load_weights_folder


def load_external_predictions(opt):
    print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
    pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.eval_eigen_to_benchmark:
        mapping_path = os.path.join(
            SPLITS_DIR,
            "benchmark",
            "eigen_to_benchmark_ids.npy",
        )
        eigen_to_benchmark_ids = np.load(mapping_path)
        pred_disps = pred_disps[eigen_to_benchmark_ids]

    return pred_disps


def save_benchmark_predictions(pred_disps, load_weights_folder):
    save_dir = os.path.join(
        load_weights_folder,
        "benchmark_predictions",
    )
    os.makedirs(save_dir, exist_ok=True)

    print("-> Saving benchmark predictions to {}".format(save_dir))

    for index, pred_disp in enumerate(pred_disps):
        resized_disp = cv2.resize(pred_disp, (1216, 352))
        depth = STEREO_SCALE_FACTOR / resized_disp
        depth = np.clip(depth, 0, 80)
        depth = np.uint16(depth * 256)
        save_path = os.path.join(save_dir, "{:010d}.png".format(index))
        cv2.imwrite(save_path, depth)


def evaluate_predictions(opt, pred_disps):
    min_depth = 1e-3
    max_depth = 150.0

    gt_depths = np.load(
        GT_DEPTHS_PATH,
        fix_imports=True,
        encoding="latin1",
    )["data"]

    if len(pred_disps) != len(gt_depths):
        raise ValueError(
            "Prediction count ({}) does not match ground-truth count ({})".format(
                len(pred_disps), len(gt_depths)
            )
        )

    print("-> Evaluating")

    if opt.eval_stereo:
        print(
            "   Stereo evaluation - disabling median scaling, "
            "scaling by {}".format(STEREO_SCALE_FACTOR)
        )
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")
        disable_median_scaling = opt.disable_median_scaling
        pred_depth_scale_factor = opt.pred_depth_scale_factor

    errors = []
    ratios = []

    for pred_disp, gt_depth in zip(pred_disps, gt_depths):
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-12)

        mask = np.logical_and(
            gt_depth > min_depth,
            gt_depth < max_depth,
        )

        if opt.eval_split == "eigen":
            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        valid_gt_depth = gt_depth[mask]

        pred_depth *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(valid_gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth = np.clip(pred_depth, min_depth, max_depth)
        errors.append(compute_errors(valid_gt_depth, pred_depth))

    if not disable_median_scaling:
        ratios = np.asarray(ratios)
        median_ratio = np.median(ratios)
        print(
            " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
                median_ratio,
                np.std(ratios / median_ratio),
            )
        )

    mean_errors = np.asarray(errors).mean(axis=0)

    print(
        "\n  "
        + ("{:>8} | " * 7).format(
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "a1",
            "a2",
            "a3",
        )
    )
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


def evaluate(opt):
    if sum((opt.eval_mono, opt.eval_stereo)) != 1:
        raise ValueError(
            "Please choose mono or stereo evaluation by setting "
            "either --eval_mono or --eval_stereo"
        )

    if opt.ext_disp_to_eval is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu"
        )
        pred_disps, load_weights_folder = load_model_predictions(opt, device)
    else:
        pred_disps = load_external_predictions(opt)
        load_weights_folder = os.path.expanduser(opt.load_weights_folder or ".")

    if opt.save_pred_disps:
        output_path = os.path.join(
            load_weights_folder,
            "disps_{}_split.npy".format(opt.eval_split),
        )
        print("-> Saving predicted disparities to {}".format(output_path))
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return

    if opt.eval_split == "benchmark":
        save_benchmark_predictions(pred_disps, load_weights_folder)
        print("-> No ground truth is available for benchmark evaluation.")
        return

    evaluate_predictions(opt, pred_disps)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
