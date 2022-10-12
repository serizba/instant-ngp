from pathlib import Path
import argparse
import tempfile
import shutil
import json
import os
import open3d as o3d
import time
import tqdm
import numpy as np
from PIL import Image

import struct
import imageio

import swn.read_write_model as rwm
from swn.data.kitti import KITTI_TEST_SEQS

import cv2


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm


def closest_point_2_lines(oa, da, ob,
						  db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c) ** 2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa + ta * da + ob + tb * db) * 0.5, denom


import pyngp as ngp  # noqa


def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:, :, :3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)


def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def write_image(file, img, quality=95):
	if os.path.splitext(file)[1] == ".bin":
		if img.shape[2] < 4:
			img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
		with open(file, "wb") as f:
			f.write(struct.pack("ii", img.shape[0], img.shape[1]))
			f.write(img.astype(np.float16).tobytes())
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			# Unmultiply alpha
			img[..., 0:3] = np.divide(img[..., 0:3], img[..., 3:4], out=np.zeros_like(img[..., 0:3]),
									  where=img[..., 3:4] != 0)
			img[..., 0:3] = linear_to_srgb(img[..., 0:3])
		else:
			img = linear_to_srgb(img)
		write_image_imageio(file, img, quality)


def main(args):
	if args.depth == 'gt' and args.pose == 'colmap':
		raise ValueError("Cannot GT depths with COLMAP poses")

	PATH_RAW_DATA = args.path_raw_data
	PATH_DEPTH_GT = args.path_depth_gt
	PATH_COLMAP = args.path_colmap
	PATH_OUTPUT = args.path_output

	for seq in KITTI_TEST_SEQS:
		print(f'Processing sequence: {seq}')

		#####################################################
		# 1. Preprocess the sequence and create scene files #
		#####################################################

		temp_dir = Path(tempfile.mkdtemp())

		scene_file = PATH_OUTPUT / f'pose_{args.pose}_depth_{args.depth}' / seq
		scene_file.mkdir(parents=True, exist_ok=True)
		scene_file = scene_file / 'transforms.json'

		# 1.1 Create scene file with the poses
		if args.pose == 'colmap':
			# Convert COLMAP model into text
			pts_model = rwm.read_points3D_binary(PATH_COLMAP / seq / 'colmap' / 'sparse' / 'points3D.bin')
			cam_model = rwm.read_cameras_binary(PATH_COLMAP / seq / 'colmap' / 'sparse' / 'cameras.bin')
			rwm.write_cameras_text(cam_model, temp_dir / 'cameras.txt')
			img_model = rwm.read_images_binary(PATH_COLMAP / seq / 'colmap' / 'sparse' / 'images.bin')
			rwm.write_images_text(img_model, temp_dir / 'images.txt')

			# Run COLMAP2NERF
			os.system(' '.join([
				'python3 scripts/colmap2nerf.py',
				'--images', str(PATH_RAW_DATA / seq[:10] / seq / 'image_02' / 'data'),
				'--text', str(temp_dir),
				'--out', str(scene_file),
			]))
		elif args.pose == 'gt':
			from swn.data.raw import raw
			import swn.data.kitti_loader as kl
			kitti = raw(PATH_RAW_DATA, seq[:10], seq.split('_')[-2], frame_range=None)
			kitti.load_oxts()
			kitti.load_rgb()

			data = {
				"camera_angle_x": 1.4351344964896717,
				"camera_angle_y": 0.49929363336706795,
				"fl_x": 711.5252890399647,
				"fl_y": 735.3929070454187,
				"k1": 0,
				"k2": 0,
				"p1": 0,
				"p2": 0,
				"cx": 621.0,
				"cy": 187.5,
				"w": 1242.0,
				"h": 375.0,
				"aabb_scale": 1,
				"frames": []
			}

			poses_imu_w, calibrations, focal = kl.get_poses_calibration(f'{PATH_RAW_DATA}/{seq[:10]}/{seq}',
																		oxts_path_tracking=None)
			velo2imu = kl.invert_transformation(calibrations[0][:3, :3], calibrations[0][:3, 3])
			poses_velo_w = np.matmul(poses_imu_w, velo2imu)
			cam_poses = kl.get_camera_poses(poses_velo_w, calibrations, [0, len(kitti.oxts) - 1])

			for i in range(len(kitti.oxts)):
				try:
					b = sharpness(kitti.rgbL_path[i])
				except:
					print(f"Error: failed to open {kitti.rgbL_path[i]}")
					continue
				frame = {"file_path": kitti.rgbL_path[i], "sharpness": b, "transform_matrix": kitti.oxts[i].T_w_imu}
				frame['transform_matrix'] = cam_poses[i * 2]
				data["frames"].append(frame)

			# # find a central point they are all looking at
			# print("computing center of attention...")
			# totw = 0.0
			# totp = np.array([0.0, 0.0, 0.0])
			# for f in data["frames"]:
			# 	mf = f["transform_matrix"][0:3, :]
			# 	for g in data["frames"]:
			# 		mg = g["transform_matrix"][0:3, :]
			# 		p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
			# 		if w > 0.00001:
			# 			totp += p * w
			# 			totw += w
			# if totw > 0.0:
			# 	totp /= totw
			# print(totp)  # the cameras are looking at totp
			# for f in data["frames"]:
			# 	f["transform_matrix"][0:3, 3] -= totp
			#
			# avglen = 0.
			# for f in data["frames"]:
			# 	avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
			# avglen /= len(data["frames"])
			# print("avg camera distance from origin", avglen)
			for f in data["frames"]:
			# 	f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
				f["transform_matrix"] = f["transform_matrix"].tolist()

			scale_gt = 1.0

		# 1.2 Modify scene file with the relative paths and depths
		if args.pose == 'colmap':
			with open(scene_file, 'r') as f:
				data = json.load(f)

		for frame in data['frames']:

			# Update the relative path
			img_path = PATH_RAW_DATA / seq[:10] / seq / 'image_02' / 'data' / frame['file_path'][-14:]
			print(img_path)
			frame['file_path'] = os.path.relpath(str(img_path), scene_file.parent.resolve())
			frame['file_path_abs'] = str(img_path)

			if args.depth == 'colmap':

				# Find the corresponding image model
				try:
					v = next(v for v in img_model.values() if v.name in img_path.name)
				except:
					print(f'Could not find depth for {img_path.name}')
					continue

				# Project the 3D points into the image
				colmap_depths = np.array(
					[(v.qvec2rotmat() @ pts_model[p3d].xyz + v.tvec)[2] for p3d in v.point3D_ids[v.point3D_ids > -1]])
				colmap_coords = np.array(
					[v.xys[np.where(v.point3D_ids == p3d)][0, ::-1] for p3d in v.point3D_ids[v.point3D_ids > -1]])
				colmap_coords = np.round(colmap_coords).astype(int)
				H, W = cam_model[1].height, cam_model[1].width
				# Filter out points outside the image
				mask = (colmap_coords[:, 1] >= 0) & (colmap_coords[:, 1] < W) & (colmap_coords[:, 0] >= 0) & (
						colmap_coords[:, 0] < H)
				colmap_coords = colmap_coords[mask]
				colmap_depths = colmap_depths[mask]

				# Fill sparse depth map
				depth_img = np.zeros((H, W), dtype=np.uint16)
				depth_img[colmap_coords[:, 0], colmap_coords[:, 1]] = (colmap_depths * 256).astype(np.uint16)

				# Save depth map
				Image.fromarray(depth_img).save(temp_dir / img_path.name)
				frame['depth_path'] = os.path.relpath(temp_dir / img_path.name, scene_file.parent.resolve())
				frame['depth_path_abs'] = str(temp_dir / img_path.name)

			elif args.depth == 'gt':

				depth_path_1 = PATH_DEPTH_GT / 'val' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / \
							   frame['file_path'].split('/')[-1]
				depth_path_2 = PATH_DEPTH_GT / 'train' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / \
							   frame['file_path'].split('/')[-1]
				if depth_path_1.is_file():
					frame['depth_path'] = os.path.relpath(depth_path_1.resolve(), scene_file.parent.resolve())
					frame['depth_path_abs'] =str(depth_path_1)
				elif depth_path_2.is_file():
					frame['depth_path_abs'] = str(depth_path_2)
					frame['depth_path'] = os.path.relpath(depth_path_2.resolve(), scene_file.parent.resolve())

		# Depth scale
		if args.depth == 'colmap':
			data['enable_depth_loading'] = True
			data['integer_depth_scale'] = 1.0 / 256.0
		elif args.depth == 'gt':
			data['enable_depth_loading'] = True
			data['integer_depth_scale'] = scale_gt * (1.0 / 256.0)

		with open(scene_file, 'w') as json_file:
			json.dump(data, json_file, indent=4)

		#################
		# 3. Visualization #
		#################
		if args.visualize:
			pointcloud = []
			pointcloud.append(o3d.geometry.TriangleMesh().create_coordinate_frame())
			for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,65, 70, 75]:
				frame = data['frames'][i]
				if 'depth_path_abs' in frame:
					rbg_path = frame['file_path_abs']
					depth_path = frame['depth_path_abs']
					print(depth_path)
					print(rbg_path)
					focal_length_x = float(data['fl_x'])
					focal_length_y = float(data['fl_y'])
					central_1 = float(data['cx'])
					central_2 = float(data['cy'])
					width = int(data['w'])
					height = int(data['h'])
					color_raw = o3d.io.read_image(rbg_path)
					depth_raw = o3d.io.read_image(depth_path)
					rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
																					depth_scale=256,
																					depth_trunc=1000,
																					convert_rgb_to_intensity=False)
					pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
																		 o3d.camera.PinholeCameraIntrinsic(width,
																										   height,
																										   focal_length_x,
																										   focal_length_y,
																										   central_1,
																										   central_2))
					transformation = frame["transform_matrix"].copy()

					pointcloud.append(o3d.geometry.TriangleMesh().create_coordinate_frame().transform(transformation))

					pcd = pcd.transform(transformation)
					pointcloud.append(pcd)
			o3d.visualization.draw_geometries(pointcloud)
	#################
	# 2. Train Nerf #
	#################

	# Args
	sharpen = 0.0
	exposure = 0.0
	n_steps = args.n_steps

	network = Path(__file__).parent / 'configs' / 'nerf' / 'base.json'
	network = network.resolve()

	mode = ngp.TestbedMode.Nerf
	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = sharpen
	testbed.exposure = exposure
	testbed.load_training_data(str(scene_file.parent))
	testbed.reload_network_from_file(str(network))
	testbed.shall_train = True
	testbed.nerf.render_with_camera_distortion = True

	old_training_step = 0
	n_steps = n_steps

	if args.depth != 'none' and testbed.nerf.training.depth_supervision_lambda == 0.0:
		raise ValueError(
			'Depth supervision is disabled but depth is provided, set a non-zero value for depth_supervision_lambda')

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm.tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	###########################
	# 2. Save Nerf renderings #
	###########################

	print('Evaluating sequence:', seq)
	with open(scene_file) as f:
		test_transforms = json.load(f)

	# Evaluate metrics on black background
	testbed.background_color = [0.0, 0.0, 0.0, 1.0]
	# Prior nerf papers don't typically do multi-sample anti aliasing.
	# So snap all pixels to the pixel centers.
	testbed.snap_to_pixel_centers = True
	spp = 8
	testbed.nerf.rendering_min_transmittance = 1e-4
	testbed.fov_axis = 0
	testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
	testbed.shall_train = False

	(scene_file.parent / 'rgb').mkdir(parents=True, exist_ok=True)
	(scene_file.parent / 'depth').mkdir(parents=True, exist_ok=True)

	H, W = int(test_transforms["h"]), int(test_transforms["w"])
	with tqdm.tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
		for i, frame in t:
			testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1, :])

			# TODO
			# DO fancy stuff with linear and srgb

			# Render and save RGB
			testbed.render_mode = ngp.Shade
			image = testbed.render(W, H, spp, True)
			write_image(str(scene_file.parent / 'rgb' / frame['file_path'].split('/')[-1]), image)

			# Render and save depth
			testbed.render_mode = ngp.Depth
			image = testbed.render(W, H, spp, True)
			np.save(str(scene_file.parent / 'depth' / frame['file_path'].split('/')[-1][:10]), image[:, :, 0])

			# For debug
			import matplotlib.pyplot as plt
			plt.imsave(str(scene_file.parent / 'depth' / frame['file_path'].split('/')[-1]), image[:, :, 0])

	# Remove temp dir
	shutil.rmtree(temp_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Run fancy stuff")
	parser.add_argument("--pose", default='gt', choices=['colmap', 'gt'], help="Use COLMAP poses or GT poses")
	parser.add_argument("--depth", default='none', choices=['none', 'colmap', 'gt'],
						help="Use COLMAP sparse depth as supervision or GT semi-dense depth")
	parser.add_argument("--n_steps", type=int, help="Number of steps in Nerf training", default=20000)
	parser.add_argument("--path_raw_data", type=Path, help="Path to raw KITTI data")
	parser.add_argument("--path_depth_gt", type=Path, help="Path to GT depth data")
	parser.add_argument("--path_colmap", type=Path, help="Path to COLMAP data")
	parser.add_argument("--path_output", type=Path, help="Path to output data")
	parser.add_argument("--visualize", action='store_true', help="Visualize pointclouds")
	main(parser.parse_args())

# python3 ablation.py --path_raw_data /home/serizba/phd/data/kitti/raw_data_2 --path_depth_gt /home/serizba/phd/data/kitti/data_depth_annotated --path_colmap /home/serizba/phd/data/kitti/eigen_reconstructions --path_output /home/serizba/phd/supervisingnerf/fork/ablation/
