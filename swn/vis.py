import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from swn.data.kitti import KITTI_TEST_SEQS


def parse_args():
	parser = argparse.ArgumentParser(description="Visualizaton")
	parser.add_argument("--path_raw_data", default="images", help="input path to the images depth and renderings")
	parser.add_argument("--path_depth_gt", default="images", help="gt path to the images depth and rgb")
	parser.add_argument("--path_output", default="test", help="Data to test")
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	from swn.data.raw import raw
	import swn.data.kitti_loader as kl
	kitti = raw(args.path_raw_data, KITTI_TEST_SEQS[0][:10], KITTI_TEST_SEQS[0].split('_')[-2], frame_range=None)
	kitti.load_oxts()
	kitti.load_rgb()
	poses_imu_w, calibrations, focal = kl.get_poses_calibration(f'{args.path_raw_data}/{KITTI_TEST_SEQS[0][:10]}/{KITTI_TEST_SEQS[0]}',
																oxts_path_tracking=None)

	velo2imu = kl.invert_transformation(calibrations[0][:3, :3], calibrations[0][:3, 3])
	poses_velo_w = np.matmul(poses_imu_w, velo2imu)
	cam_poses = kl.get_camera_poses(poses_velo_w, calibrations, [0, len(kitti.oxts) - 1])
	data = {'frames':[]}
	for i in range(len(kitti.oxts)):
		print(kitti.rgbL_path[i])
		frame = {"file_path": kitti.rgbL_path[i]}
		frame["transform_matrix"]= kitti.oxts[i].T_w_imu # ????
		frame['transform_matrix'] = cam_poses[i * 2]
		data["frames"].append(frame)
	PATH_DEPTH_GT = Path('/home/rene/Kitti/data_depth_annotated')
	for frame in data['frames']:
		depth_path_1 = PATH_DEPTH_GT / 'val' / KITTI_TEST_SEQS[0] / 'proj_depth' / 'groundtruth' / 'image_02' / \
					   frame['file_path'].split('/')[-1]
		depth_path_2 = PATH_DEPTH_GT / 'train' / KITTI_TEST_SEQS[0] / 'proj_depth' / 'groundtruth' / 'image_02' / \
					   frame['file_path'].split('/')[-1]
		if depth_path_1.is_file():
			frame['depth_path'] = str(depth_path_1.absolute())
		elif depth_path_2.is_file():
			frame['depth_path'] = str(depth_path_2.absolute())

	pointcloud = []
	pointcloud.append(o3d.geometry.TriangleMesh().create_coordinate_frame())
	for i in [5, 10, 15, 20, 25]:
		frame = data['frames'][i]
		if 'depth_path' in frame:
			rbg_path = frame['file_path']
			depth_path = frame['depth_path']

			focal_length = 721.5377
			central_1 = 621.0
			central_2 = 187.5
			color_raw = o3d.io.read_image(rbg_path)
			depth_raw = o3d.io.read_image(depth_path)
			rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,depth_scale=256,depth_trunc=80,convert_rgb_to_intensity=False)
			pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
																 o3d.camera.PinholeCameraIntrinsic(1242, 375, focal_length,
																								   focal_length, central_1,
																								   central_2))
			transformation = frame["transform_matrix"].copy()
			#FIX TRANSFORMATION transformation[0][3] = frame["transform_matrix"][0][3]
			transformation  = np.linalg.inv(transformation)

			pointcloud.append(o3d.geometry.TriangleMesh().create_coordinate_frame().transform(transformation))

			pcd = pcd.transform(transformation)
			pointcloud.append(pcd)
	o3d.visualization.draw_geometries(pointcloud)
