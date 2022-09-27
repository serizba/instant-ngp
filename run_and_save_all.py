from pathlib import Path
import argparse
import tempfile
import shutil
import json
import os

import time
import tqdm
import numpy as np
from PIL import Image

import struct
import imageio

import swn.read_write_model as rwm
from swn.data.kitti import KITTI_TEST_SEQS

import pyngp as ngp # noqa


def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
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
            img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
            img[...,0:3] = linear_to_srgb(img[...,0:3])
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
            raise NotImplementedError('GT poses not implemented yet')

        # 1.2 Modify scene file with the relative paths and depths
        with open(scene_file, 'r') as f:
            data = json.load(f)

        for frame in data['frames']:

            # Update the relative path
            img_path = PATH_RAW_DATA / seq[:10] / seq / 'image_02' / 'data' / frame['file_path'][-14:]
            frame['file_path'] = os.path.relpath(str(img_path), scene_file.parent.resolve())

            if args.depth == 'colmap':

                # Find the corresponding image model
                try:
                    v = next(v for v in img_model.values() if v.name in img_path.name)
                except:
                    print(f'Could not find depth for {img_path.name}')
                    continue

                # Project the 3D points into the image
                colmap_depths = np.array([(v.qvec2rotmat() @ pts_model[p3d].xyz + v.tvec)[2] for p3d in v.point3D_ids[v.point3D_ids > -1]])
                colmap_coords = np.array([v.xys[np.where(v.point3D_ids == p3d)][0, ::-1] for p3d in v.point3D_ids[v.point3D_ids > -1]])
                colmap_coords = np.round(colmap_coords).astype(int)

                # Fill sparse depth map
                depth_img = np.zeros((cam_model[1].height, cam_model[1].width), dtype=np.uint16)
                depth_img[colmap_coords[:, 0], colmap_coords[:, 1]] = (colmap_depths * 256).astype(np.uint16)

                # Save depth map
                Image.fromarray(depth_img).save(temp_dir / img_path.name)
                frame['depth_path'] = os.path.relpath(temp_dir / img_path.name, scene_file.parent.resolve())                

            elif args.depth == 'gt':
                depth_path_1 = PATH_DEPTH_GT / 'val' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / frame['file_path'].split('/')[-1]
                depth_path_2 = PATH_DEPTH_GT / 'train' / seq / 'proj_depth' / 'groundtruth' / 'image_02' / frame['file_path'].split('/')[-1]
                if depth_path_1.is_file():
                    frame['depth_path'] = os.path.relpath(depth_path_1.resolve(), scene_file.parent.resolve())
                elif depth_path_2.is_file():
                    frame['depth_path'] = os.path.relpath(depth_path_2.resolve(), scene_file.parent.resolve())

        # Depth scale
        if args.depth == 'colmap':
            data['enable_depth_loading'] = True
            data['integer_depth_scale'] = 1.0 / 256.0
        
        with open(scene_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)

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
            raise ValueError('Depth supervision is disabled but depth is provided, set a non-zero value for depth_supervision_lambda')

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
              
                testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])

                # TODO
                # DO fancy stuff with linear and srgb

                # Render and save RGB
                testbed.render_mode = ngp.Shade
                image = testbed.render(W, H, spp, True)
                write_image(str(scene_file.parent / 'rgb' / frame['file_path'].split('/')[-1]), image)

                # Render and save depth
                testbed.render_mode = ngp.Depth
                image = testbed.render(W, H, spp, True)
                np.save(str(scene_file.parent / 'depth' / frame['file_path'].split('/')[-1][:10]), image[:,:,0])

                # For debug
                import matplotlib.pyplot as plt
                plt.imsave(str(scene_file.parent / 'depth' / frame['file_path'].split('/')[-1]), image[:,:,0])


        # Remove temp dir
        shutil.rmtree(temp_dir)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run fancy stuff")
    parser.add_argument("--pose", default='gt', choices=['colmap', 'gt'], help="Use COLMAP poses or GT poses")
    parser.add_argument("--depth", default='none', choices=['none', 'colmap', 'gt'], help="Use COLMAP sparse depth as supervision or GT semi-dense depth")
    parser.add_argument("--n_steps", type=int, help="Number of steps in Nerf training", default=20000)
    parser.add_argument("--path_raw_data", type=Path, help="Path to raw KITTI data")
    parser.add_argument("--path_depth_gt", type=Path, help="Path to GT depth data")
    parser.add_argument("--path_colmap", type=Path, help="Path to COLMAP data")
    parser.add_argument("--path_output", type=Path, help="Path to output data")
    main(parser.parse_args())


# python3 ablation.py --path_raw_data /home/serizba/phd/data/kitti/raw_data_2 --path_depth_gt /home/serizba/phd/data/kitti/data_depth_annotated --path_colmap /home/serizba/phd/data/kitti/eigen_reconstructions --path_output /home/serizba/phd/supervisingnerf/fork/ablation/