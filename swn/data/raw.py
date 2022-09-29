"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import json
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

import sys
sys.path.append('/home/serizba/phd/supervisingnerf/fork/')

from scripts.colmap2nerf import sharpness, closest_point_2_lines

from swn.data import kitti_loader as kl


class raw:
	"""Load and parse raw data into a usable format."""

	def __init__(self, base_path, date, drive, frame_range=None):
		"""Set the path."""
		self.drive = date + '_drive_' + drive + '_sync'
		self.calib_path = os.path.join(base_path, date)
		self.data_path = os.path.join(base_path, date, self.drive)
		self.frame_range = frame_range

	def _load_calib_rigid(self, filename):
		"""Read a rigid transform calibration file as a numpy.array."""
		filepath = os.path.join(self.calib_path, filename)
		data = self.read_calib_file(filepath)
		return self.transform_from_rot_trans(data['R'], data['T'])

	def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
		# We'll return the camera calibration as a dictionary
		data = {}

		# Load the rigid transformation from velodyne coordinates
		# to unrectified cam0 coordinates
		T_cam0unrect_velo = self._load_calib_rigid(velo_to_cam_file)

		# Load and parse the cam-to-cam calibration data
		cam_to_cam_filepath = os.path.join(self.calib_path, cam_to_cam_file)
		filedata = self.read_calib_file(cam_to_cam_filepath)

		# Create 3x4 projection matrices
		P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
		P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
		P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
		P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

		# Create 4x4 matrix from the rectifying rotation matrix
		R_rect_00 = np.eye(4)
		R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))

		# Compute the rectified extrinsics from cam0 to camN
		T0 = np.eye(4)
		T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
		T1 = np.eye(4)
		T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
		T2 = np.eye(4)
		T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
		T3 = np.eye(4)
		T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

		# Compute the velodyne to rectified camera coordinate transforms
		data['T_cam0_velo'] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
		data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
		data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
		data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))

		# Compute the camera intrinsics
		data['K_cam0'] = P_rect_00[0:3, 0:3]
		data['K_cam1'] = P_rect_10[0:3, 0:3]
		data['K_cam2'] = P_rect_20[0:3, 0:3]
		data['K_cam3'] = P_rect_30[0:3, 0:3]

		# Compute the stereo baselines in meters by projecting the origin of
		# each camera frame into the velodyne frame and computing the distances
		# between them
		p_cam = np.array([0, 0, 0, 1])
		p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
		p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
		p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
		p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

		data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
		data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

		return data

	def load_calib(self):
		"""Load and compute intrinsic and extrinsic calibration parameters."""
		# We'll build the calibration parameters as a dictionary, then
		# convert it to a namedtuple to prevent it from being modified later
		data = {}

		# Load the rigid transformation from velodyne to IMU
		data['T_velo_imu'] = self._load_calib_rigid('calib_imu_to_velo.txt')

		# Load the camera intrinsics and extrinsics
		data.update(self._load_calib_cam_to_cam(
			'calib_velo_to_cam.txt', 'calib_cam_to_cam.txt'))

		# Pre-compute the IMU to rectified camera coordinate transforms
		data['T_cam0_imu'] = data['T_cam0_velo'].dot(data['T_velo_imu'])
		data['T_cam1_imu'] = data['T_cam1_velo'].dot(data['T_velo_imu'])
		data['T_cam2_imu'] = data['T_cam2_velo'].dot(data['T_velo_imu'])
		data['T_cam3_imu'] = data['T_cam3_velo'].dot(data['T_velo_imu'])

		self.calib = namedtuple('CalibData', data.keys())(*data.values())

	def load_timestamps(self):
		"""Load timestamps from file."""
		print('Loading OXTS timestamps from ' + self.drive + '...')

		timestamp_file = os.path.join(
			self.data_path, 'oxts', 'timestamps.txt')

		# Read and parse the timestamps
		self.timestamps = []
		with open(timestamp_file, 'r') as f:
			for line in f.readlines():
				# NB: datetime only supports microseconds, but KITTI timestamps
				# give nanoseconds, so need to truncate last 4 characters to
				# get rid of \n (counts as 1) and extra 3 digits
				t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
				self.timestamps.append(t)

		# Subselect the chosen range of frames, if any
		if self.frame_range:
			self.timestamps = [self.timestamps[i] for i in self.frame_range]

		print('Found ' + str(len(self.timestamps)) + ' timestamps...')

		print('done.')

	def _poses_from_oxts(self, oxts_packets):
		"""Helper method to compute SE(3) pose matrices from OXTS packets."""
		er = 6378137.  # earth radius (approx.) in meters

		# compute scale from first lat value
		scale = np.cos(oxts_packets[0].lat * np.pi / 180.)

		t_0 = []  # initial position
		poses = []  # list of poses computed from oxts
		for packet in oxts_packets:
			# Use a Mercator projection to get the translation vector
			tx = scale * packet.lon * np.pi * er / 180.
			ty = scale * er * \
				 np.log(np.tan((90. + packet.lat) * np.pi / 360.))
			tz = packet.alt
			t = np.array([tx, ty, tz])

			# We want the initial position to be the origin, but keep the ENU
			# coordinate system
			if len(t_0) == 0:
				t_0 = t

			# Use the Euler angles to get the rotation matrix
			Rx = self.rotx(packet.roll)
			Ry = self.roty(packet.pitch)
			Rz = self.rotz(packet.yaw)
			R = Rz.dot(Ry.dot(Rx))

			# Combine the translation and rotation into a homogeneous transform
			poses.append(self.transform_from_rot_trans(R, t - t_0))

		return poses

	def load_oxts(self):
		"""Load OXTS data from file."""
		print('Loading OXTS data from ' + self.drive + '...')

		# Find all the data files
		oxts_path = os.path.join(self.data_path, 'oxts', 'data', '*.txt')
		oxts_files = sorted(glob.glob(oxts_path))

		# Subselect the chosen range of frames, if any
		if self.frame_range:
			oxts_files = [oxts_files[i] for i in self.frame_range]

		print('Found ' + str(len(oxts_files)) + ' OXTS measurements...')

		# Extract the data from each OXTS packet
		# Per dataformat.txt
		OxtsPacket = namedtuple('OxtsPacket',
								'lat, lon, alt, ' +
								'roll, pitch, yaw, ' +
								'vn, ve, vf, vl, vu, ' +
								'ax, ay, az, af, al, au, ' +
								'wx, wy, wz, wf, wl, wu, ' +
								'pos_accuracy, vel_accuracy, ' +
								'navstat, numsats, ' +
								'posmode, velmode, orimode')

		oxts_packets = []
		for filename in oxts_files:
			with open(filename, 'r') as f:
				for line in f.readlines():
					line = line.split()
					# Last five entries are flags and counts
					line[:-5] = [float(x) for x in line[:-5]]
					line[-5:] = [float(x) for x in line[-5:]]

					data = OxtsPacket(*line)
					oxts_packets.append(data)

		# Precompute the IMU poses in the world frame
		T_w_imu = self._poses_from_oxts(oxts_packets)

		# Bundle into an easy-to-access structure
		OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
		self.oxts = []
		for (p, T) in zip(oxts_packets, T_w_imu):
			self.oxts.append(OxtsData(p, T))

		print('done.')

	def rotx(self, t):
		"""Rotation about the x-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[1, 0, 0],
						 [0, c, -s],
						 [0, s, c]])

	def roty(self, t):
		"""Rotation about the y-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, 0, s],
						 [0, 1, 0],
						 [-s, 0, c]])

	def rotz(self, t):
		"""Rotation about the z-axis."""
		c = np.cos(t)
		s = np.sin(t)
		return np.array([[c, -s, 0],
						 [s, c, 0],
						 [0, 0, 1]])

	def transform_from_rot_trans(self, R, t):
		"""Transforation matrix from rotation matrix and translation vector."""
		R = R.reshape(3, 3)
		t = t.reshape(3, 1)
		return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

	def read_calib_file(self, filepath):
		"""Read in a calibration file and parse into a dictionary."""
		data = {}

		with open(filepath, 'r') as f:
			for line in f.readlines():
				key, value = line.split(':', 1)
				# The only non-float values in these files are dates, which
				# we don't care about anyway
				try:
					data[key] = np.array([float(x) for x in value.split()])
				except ValueError:
					pass

		return data

	def load_rgb(self):
		"""Load RGB stereo images from file.
		Setting imformat='cv2' will convert the images to uint8 and BGR for
		easy use with OpenCV.
		"""
		print('Loading color images from ' + self.drive + '...')

		imL_path = os.path.join(self.data_path, 'image_02', 'data', '*.png')
		imR_path = os.path.join(self.data_path, 'image_03', 'data', '*.png')

		imL_files = sorted(glob.glob(imL_path))
		imR_files = sorted(glob.glob(imR_path))

		# Subselect the chosen range of frames, if any
		if self.frame_range:
			imL_files = [imL_files[i] for i in self.frame_range]
			imR_files = [imR_files[i] for i in self.frame_range]

		print('Found ' + str(len(imL_files)) + ' image pairs...')

		self.rgb = self.load_stereo_pairs(imL_files, imR_files)
		self.rgbL_path = imL_files
		self.rgbR_path = imR_files
		print('done.')

	def load_stereo_pairs(self,imL_files, imR_files):
		"""Helper method to read stereo image pairs."""
		StereoPair = namedtuple('StereoPair', 'left, right')

		impairs = []
		for imfiles in zip(imL_files, imR_files):
			try:
				imL = np.uint8(mpimg.imread(imfiles[0]) * 255)
				imR = np.uint8(mpimg.imread(imfiles[1]) * 255)
				impairs.append(StereoPair(imL, imR))
			except:
				print('Could not read image pair: ' + str(imfiles))
				continue

		return impairs


if __name__ == "__main__":
	kitti = raw("/home/serizba/phd/data/kitti/raw_data_2/", "2011_09_26", "0002", frame_range=None)
	kitti.load_oxts()
	kitti.load_rgb()

	out = {
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
		"aabb_scale": 16,
		"frames":[]
	}

	poses_imu_w, calibrations, focal = kl.get_poses_calibration('/home/serizba/phd/data/kitti/raw_data_2/2011_09_26/2011_09_26_drive_0002_sync', oxts_path_tracking=None)
	imu2velo, velo2cam, c2leftRGB, c2rightRGB, _ = calibrations
	velo2imu = kl.invert_transformation(calibrations[0][:3, :3], calibrations[0][:3, 3])
	poses_velo_w = np.matmul(poses_imu_w, velo2imu)
	cam_poses = kl.get_camera_poses(poses_velo_w, calibrations, [0, 76])

	for i in range(len(kitti.oxts)):
		try:
			b = sharpness(kitti.rgbL_path[i])
		except:
			print(f"Error: failed to open {kitti.rgbL_path[i]}")
			continue
		frame = {"file_path": os.path.relpath(kitti.rgbL_path[i], '/home/serizba/phd/supervisingnerf/fork/ablation/transforms_gt/'), "sharpness": b, "transform_matrix": kitti.oxts[i].T_w_imu}
		frame['transform_matrix'] = cam_poses[i * 2]
		out["frames"].append(frame)


	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = f["transform_matrix"][0:3,:]
		for g in out["frames"]:
			mg = g["transform_matrix"][0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.00001:
				totp += p*w
				totw += w
	if totw > 0.0:
		totp /= totw
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= totp

	avglen = 0.
	for f in out["frames"]:
		avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
	avglen /= 77
	print("avg camera distance from origin", avglen)
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
		PATH_DEPTH_GT = Path('/home/serizba/phd/data/kitti/data_depth_annotated')
		seq = '2011_09_26_drive_0002_sync'
		depth_path_1 = PATH_DEPTH_GT / 'val' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / f['file_path'].split('/')[-1]
		depth_path_2 = PATH_DEPTH_GT / 'train' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / f['file_path'].split('/')[-1]
		print(depth_path_1)
		if depth_path_1.is_file():
			print('DEPTH FOUND')
			f['depth_path'] = os.path.relpath(depth_path_1.resolve(), '/home/serizba/phd/supervisingnerf/fork/ablation/transforms_gt')
		elif depth_path_2.is_file():
			print('DEPTH FOUND')
			f['depth_path'] = os.path.relpath(depth_path_2.resolve(), '/home/serizba/phd/supervisingnerf/fork/ablation/transforms_gt')

	
	out['enable_depth_loading'] = True
	out['integer_depth_scale'] = (4.0 / avglen) * (1.0 / 256.0 )
		

	with open("/home/serizba/phd/supervisingnerf/fork/ablation/transforms_gt/transforms.json", "w") as outfile:
		json.dump(out, outfile, indent=2)
