import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scripts.common import compute_error
from swn.data import KITTI
from swn.metrics import KITTIMetrics


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluation of performance for Depth and Image Renderings")
	parser.add_argument("--path_raw_data", default="images", help="input path to the images depth and renderings")
	parser.add_argument("--path_depth_gt", default="images", help="gt path to the images depth and rgb")
	parser.add_argument("--path_output", default="test", help="Data to test")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	metrics = {"abs_rel": [],
			   "sq_rel": [],
			   "rmse": [],
			   "rmse_log": [],
			   "a1": [],
			   "a2": [],
			   "a3": [],
			   "ssim": []
			   }
	kittimetrics = KITTIMetrics()
	for batch_idx, (x, y, x_nerf, y_nerf,colmap_depth) in tqdm(enumerate(
		KITTI(args.path_depth_gt, args.path_raw_data, split_type="eigen_with_gt", split="test",
			  path_nerf_renders=args.path_output))):
		errors = kittimetrics(y.unsqueeze(0), y_nerf.unsqueeze(0))
		ssim = float(compute_error("SSIM", x, x_nerf))
		plt.imsave("../test/error/" + str(batch_idx) + ".png", (torch.abs(y.squeeze() - y_nerf.squeeze()*errors['scale']) * (y.squeeze() > 0)).numpy())
		plt.imsave("../test/error/" +"gt"+ str(batch_idx) + ".png", (y.squeeze()) * (y.squeeze() > 0).numpy())
		plt.imsave("../test/error/" +"nerf"+ str(batch_idx) + ".png", (y_nerf.squeeze()*errors['scale']) * (y.squeeze() > 0).numpy())
		plt.imsave("../test/error/" +"colmap"+ str(batch_idx) + ".png", (colmap_depth.squeeze()) * (y.squeeze() > 0).numpy())
		plt.imsave("../test/error/" +"original_colmap"+ str(batch_idx) + ".png", (colmap_depth.squeeze()).numpy())
		plt.imsave("../test/error/" +"original_nerf"+ str(batch_idx) + ".png", (y_nerf.squeeze()).numpy())

		metrics['ssim'].append(ssim)
		metrics['abs_rel'].append(errors['abs_rel'])
		metrics['sq_rel'].append(errors['sq_rel'])
		metrics['rmse'].append(errors['rmse'])
		metrics['rmse_log'].append(errors['rmse_log'])
		metrics['a1'].append(errors['a1'])
		metrics['a2'].append(errors['a2'])
		metrics['a3'].append(errors['a3'])

	print("{} {} {} {} {} {} {} {}".format(np.array(metrics['ssim']).mean(), np.array(metrics['rmse']).mean(),
										   np.array(metrics['abs_rel']).mean(), np.array(metrics['sq_rel']).mean(),
										   np.array(metrics['rmse_log']).mean(), np.array(metrics['a1']).mean(),
										   np.array(metrics['a2']).mean(), np.array(metrics['a3']).mean()))
	print("ssim:{}".format(np.array(metrics['ssim']).mean()))
	print("Rmse:{}".format(np.array(metrics['rmse']).mean()))
	print("Abs_rel:{}".format(np.array(metrics['abs_rel']).mean()))
	print("Sq_rel:{}".format(np.array(metrics['sq_rel']).mean()))
	print("Rmse_log:{}".format(np.array(metrics['rmse_log']).mean()))
	print("A1:{}".format(np.array(metrics['a1']).mean()))
	print("A2:{}".format(np.array(metrics['a2']).mean()))
	print("A3:{}".format(np.array(metrics['a3']).mean()))
