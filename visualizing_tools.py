import os
import os.path as osp

import cv2
import numpy as np
import torch
import trimesh
from icecream import ic

from lib.config.visualizing_config import parse_config
from lib.core.camera import create_camera
from lib.core.smpl_mmvp import SMPL_MMVP
from lib.utils.color_utils import modelRender
from lib.utils.insole_utils import insole_gen


def main(**args):

    # init common param
    demo_basdir = args.pop('basdir')
    demo_dsname = args.pop('dataset')
    demo_subids = args.pop('sub_ids')
    demo_seqname = args.pop('seq_name')
    demo_frame_idx = args.pop('frame_idx')
    demo_essential_root = args.pop('essential_root')

    # create output folders
    output_folder = osp.expandvars(args.pop('output_dir'))

    # get device and set dtype
    dtype = torch.float32
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not osp.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for f in ['insoles_gt', 'meshes_gt', 'depths_gt', 'renders_gt']:
        curr_output_folder = osp.join(output_folder, f, demo_dsname,
                                      demo_subids, demo_seqname)
        os.makedirs(curr_output_folder, exist_ok=True)
        print(f'{f} will be saved in {curr_output_folder}')

    image_dir = osp.join(demo_basdir, 'images', demo_dsname, demo_subids,
                         demo_seqname)
    rgb_path = osp.join(image_dir, 'color', f'{demo_frame_idx:03d}.jpg')
    depth_path = osp.join(image_dir, 'depth', f'{demo_frame_idx:03d}.png')
    dmask_path = osp.join(image_dir, 'depth_mask', f'{demo_frame_idx:03d}.jpg')
    insole_path = osp.join(image_dir, 'insole', f'{demo_frame_idx:03d}.npy')

    smpl_gt_path = osp.join(demo_basdir, 'annotations', demo_dsname,
                            'smpl_pose', demo_subids, demo_seqname,
                            f'smpl_{demo_frame_idx:03d}.npz')

    assert osp.exists(rgb_path), print(f'{rgb_path} not exist')
    assert osp.exists(depth_path), print(f'{depth_path} not exist')
    assert osp.exists(dmask_path), print(f'{dmask_path} not exist')
    assert osp.exists(insole_path), print(f'{insole_path} not exist')
    assert osp.exists(smpl_gt_path), print(f'{smpl_gt_path} not exist')
    # assert osp.exists(floor_info_path), print(f'{insole_path} not exist')

    rgb_image = cv2.imread(rgb_path).astype(np.float32)
    depth_map = cv2.imread(depth_path, -1) / 1000.
    dmask = np.mean(cv2.imread(dmask_path), axis=-1)
    insole_data = dict(np.load(insole_path, allow_pickle=True).item())

    # init smpl mmvp
    body_model = SMPL_MMVP(essential_root=demo_essential_root,
                           gender=args.pop('model_gender'),
                           stage='init_shape',
                           dtype=dtype).to(device)

    # Create the camera object
    rgbd_cam = create_camera(
        basdir=demo_basdir,
        dataset_name=demo_dsname,
        sub_ids=demo_subids,
        seq_name=demo_seqname,
    ).to(device=device)

    # mesh generation
    smpl_gt = dict(np.load(smpl_gt_path).items())
    params_dict = {}
    params_dict['body_pose'] = torch.from_numpy(smpl_gt['body_pose']).to(
        dtype=dtype, device=device)
    params_dict['global_orient'] = torch.from_numpy(smpl_gt['global_rot']).to(
        dtype=dtype, device=device)
    params_dict['transl'] = torch.from_numpy(smpl_gt['transl']).to(
        dtype=dtype, device=device)
    params_dict['betas'] = torch.from_numpy(smpl_gt['shape']).to(dtype=dtype,
                                                                 device=device)
    params_dict['model_scale_opt'] = torch.from_numpy(
        smpl_gt['model_scale_opt']).to(dtype=dtype, device=device)
    body_model.setPose(**params_dict)
    body_model.update_shape()
    body_model.init_plane()
    model_output = body_model.update_pose()

    # smpl gt
    vertices = model_output.vertices
    vertices_depth = rgbd_cam.tranform3d(points=vertices, type='f2d')
    live_mesh_depth = trimesh.Trimesh(
        vertices=vertices_depth.detach().cpu().numpy(),
        faces=body_model.faces,
        process=False)
    vertices_color = rgbd_cam.tranform3d(points=vertices, type='f2c')
    live_mesh_color = trimesh.Trimesh(
        vertices=vertices_color.detach().cpu().numpy(),
        faces=body_model.faces,
        process=False)
    live_mesh_depth.export(
        osp.join(output_folder, 'meshes_gt', demo_dsname, demo_subids,
                 demo_seqname, f'{demo_frame_idx:03d}.obj'))

    # depth gt
    gt_depth_vmap, _, _, _ = rgbd_cam.preprocessDepth(depth_map, dmask)
    trimesh.Trimesh(vertices=gt_depth_vmap.detach().cpu().numpy()).export(
        osp.join(output_folder, 'depths_gt', demo_dsname, demo_subids,
                 demo_seqname, f'{demo_frame_idx:03d}.obj'))

    # color
    renderer = modelRender(rgbd_cam.cIntr, img_W=1280, img_H=720)

    render_result, _ = renderer.render(live_mesh_color, img=rgb_image)
    render_path = osp.join(output_folder, 'renders_gt', demo_dsname,
                           demo_subids, demo_seqname,
                           f'{demo_frame_idx:03d}.jpg')
    cv2.imwrite(render_path, render_result)

    # insole gt
    insole_gen(insole_data=insole_data['insole'],
               output_path=osp.join(output_folder, 'insoles_gt', demo_dsname,
                                    demo_subids, demo_seqname,
                                    f'{demo_frame_idx:03d}.jpg'))


if __name__ == '__main__':
    args = parse_config()
    ic(args)
    main(**args)
