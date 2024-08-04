import os
import os.path as osp
import numpy as np
import argparse
import pickle
import json
import cv2
from tqdm import tqdm
import time
import random
import shutil  
import trimesh
import imageio
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.utils.utils_mesh import flip_thetas_batch
from lib.data.dataset_wild import WildDetDataset
# from lib.model.loss import *
from lib.model.model_mesh_pvcp import MeshRegressor
from lib.utils.vismo import render_and_save, motion2video_mesh
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mesh/MB_ft_pvcp.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='myoutput/mycheckpoint/FT_pvcp_release/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, default = 'myoutput/qualitation_pvcp/mb_input_det_pose.json', help='alphapose detection result json path')
    # parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--ref_3d_motion_path', type=str, default=None, help='3D motion path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--compare', default="myoutput/qualitation_pvcp/my_compare",  help='compare 2d_pose and mesh')
    parser.add_argument('--split', default="test", help='dataset split choose')

    opts = parser.parse_args()
    return opts

def err(p, x, y):
    return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()

def solve_scale(x, y):
    print('Estimating camera transformation.')
    best_res = 100000
    best_scale = None
    for init_scale in tqdm(range(0,2000,5)):
        p0 = [init_scale, 0.0, 0.0, 0.0]
        est = least_squares(err, p0, args = (x.reshape(-1,3), y.reshape(-1,3)))
        if est['fun'] < best_res:
            best_res = est['fun']
            best_scale = est['x'][0]
    print('Pose matching error = %.2f mm.' % best_res)
    return best_scale

opts = parse_args()
args = get_config(opts.config)

# root_rel
# args.rootrel = True

smpl = SMPL(args.data_root, batch_size=1).cuda()
J_regressor = smpl.J_regressor_h36m

end = time.time()
model_backbone = load_backbone(args)
print(f'init backbone time: {(time.time()-end):02f}s')
end = time.time()
model = MeshRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout)
print(f'init whole model time: {(time.time()-end):02f}s')

if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()

chk_filename = opts.evaluate if opts.evaluate else opts.resume
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()

testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
}
# ---------------------------------Batch processing----------------------------------------------

frame_folder = "data/dataset_pvcp/frame"
image_folder = "data/dataset_pvcp/image"
video_folder = "data/dataset_pvcp/video"
train_test_list = "data/mesh/train_test_seq_id_list.json"

json_data = json.load(open(opts.json_path))
idx_groups = get_idx_groups(json_data)

if opts.split=="all":
    split_list = json.load(open(train_test_list))
else:
    split_list = json.load(open(train_test_list))[opts.split]

data_mesh = {}
# delete_directory(osp.join(opts.compare, opts.split))
for idx_key, idx_val in sorted(idx_groups.items()):
    
    verts_all = []
    reg3d_all = []
    shape_all = []
    pose_all= []
    
    print(f"-------------------idx={idx_key}/{len(idx_groups)} {len(idx_val)}-------------------------")
    # if idx_key not in split_list:continue
    # if idx_key !=92:continue
    frame_id_list = [int(frame.split(".")[0].split("_")[-1]) for frame, label in idx_val]
    consecutive_frame_list, consecutive_frame_idx = find_longest_consecutive_subset_with_gap(frame_id_list,gap=2)
    idx_val = [idx_val[i] for i in consecutive_frame_idx]
    opts.vid_path = osp.join(video_folder, sorted(os.listdir(video_folder))[idx_key])
    opts.focus = idx_key
    
    vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            start_time = time.time()
            print(batch_input.shape)
            batch_size, clip_frames = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda().float()
            
            output,out_class = model(batch_input) 
            batch_input_flip = flip_data(batch_input)
            output_flip, out_flip_class = model(batch_input_flip)
            output_flip_pose = output_flip[0]['theta'][:, :, :72]
            output_flip_shape = output_flip[0]['theta'][:, :, 72:]
            output_flip_pose = flip_thetas_batch(output_flip_pose)
            output_flip_pose = output_flip_pose.reshape(-1, 72)
            output_flip_shape = output_flip_shape.reshape(-1, 10)
            output_flip_smpl = smpl(
                betas=output_flip_shape,
                body_pose=output_flip_pose[:, 3:],
                global_orient=output_flip_pose[:, :3],
                pose2rot=True
            )
            
            output_flip_verts = output_flip_smpl.vertices.detach()
            J_regressor_batch = J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
            output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)  # (NT,17,3) 
            output_flip_back = [{
                'verts': output_flip_verts.reshape(batch_size, clip_frames, -1, 3) * 1000.0,
                'kp_3d': output_flip_kp3d.reshape(batch_size, clip_frames, -1, 3),
            }]
            output_final = [{}]
            for k, v in output_flip_back[0].items():
                output_final[0][k] = (output[0][k] + output_flip_back[0][k]) / 2.0
            output = output_final
            
            end_time = time.time()
            processing_time = (end_time - start_time) / clip_frames
            print(processing_time)
            
            verts_all.append(output[0]['verts'].cpu().numpy())
            reg3d_all.append(output[0]['kp_3d'].cpu().numpy())
            shape_all.append(output_flip_shape.cpu().numpy())
            pose_all.append(output_flip_pose.cpu().numpy())
                
    verts_all = np.hstack(verts_all)
    verts_all = np.concatenate(verts_all)
    reg3d_all = np.hstack(reg3d_all)
    reg3d_all = np.concatenate(reg3d_all)
    
    
    shape_all = np.vstack(shape_all)
    pose_all = np.vstack(pose_all)

    if opts.ref_3d_motion_path:
        ref_pose = np.load(opts.ref_3d_motion_path)
        x = ref_pose - ref_pose[:, :1]
        y = reg3d_all - reg3d_all[:, :1]
        scale = solve_scale(x, y)
        root_cam = ref_pose[:, :1] * scale
        verts_all = verts_all - reg3d_all[:,:1] + root_cam

    render_and_save(verts_all, osp.join(opts.out_path, opts.split, f'{idx_key}_mesh.mp4'), keep_imgs=False, fps=fps_in, draw_face=True, vid_size=vid_size)
    

    tmp_img_path = osp.join(opts.compare, opts.split, f"tmp_img_{idx_key}")
    tmp_vid_path = osp.join(opts.compare, opts.split, f"tmp_vid_{idx_key}")
    compare_video_path = osp.join(opts.compare,opts.split,"compare_video")
    os.makedirs(tmp_img_path, exist_ok=True)
    os.makedirs(tmp_vid_path, exist_ok=True)
    os.makedirs(compare_video_path, exist_ok=True)
    
    idx_frame_list = get_idx_frame_list(idx_groups, idx_key)
    for index, frame in enumerate(idx_frame_list):
        data_mesh[frame] = {}
        data_mesh[frame]["shape"]  = shape_all[index]
        data_mesh[frame]["pose"]  = pose_all[index]
        data_mesh[frame]["keypoint_3d_smpl"] = reg3d_all[index]
        data_mesh[frame]["verts"] = verts_all[index]
        
        mesh_folder = osp.join(opts.out_path, 'obj', f'{idx_key}')
        os.makedirs(mesh_folder, exist_ok=True)
        mesh_filename=osp.join(mesh_folder, f"{frame.split('.')[0]}.obj")
        
        # 在您的代码中保存顶点数据为.obj文件的部分之后
        if mesh_filename is not None:
            # 创建一个Trimesh对象，使用从SMPL估计模型获得的顶点数据和固定的面数据（faces）
            mesh_to_save = trimesh.Trimesh(vertices=data_mesh[frame]["verts"]*0.001, faces=smpl.faces)
            mesh_to_save.export(mesh_filename)       

        input_label = idx_groups[idx_key][index][-1]
        keypoint = [point[:2] for point in input_label["keypoints"]]
        img_path = osp.join(frame_folder, frame)

        if len(keypoint)>=10:  
            kp_img = vis_keypoint_in_img(img_path, keypoint, point_color=(0, 128, 255), skeleton_colors=(0, 255, 0), str_text=f'{idx_key}_{frame}')
        else:
            kp_img = vis_keypoint_in_img(img_path, keypoint, point_color=(0, 128, 255), skeleton_colors=(0, 255, 0), str_text=f'keupoint_num < 10')
    
        cv2.imwrite(osp.join(tmp_img_path, frame), kp_img)
    
    images_to_video(tmp_img_path, 
                    osp.join(tmp_vid_path, f'{idx_key}_j2d.mp4'),
                    frame_rate=fps_in, video_size=vid_size)
    concatenate_videos_horizontally(osp.join(tmp_vid_path, f'{idx_key}_j2d.mp4'),
                                    osp.join(opts.out_path, opts.split, f'{idx_key}_mesh.mp4'),
                                    osp.join(compare_video_path, f'{idx_key}_compare.mp4'))
    
    # delete_directory(tmp_img_path)
    # delete_directory(tmp_vid_path)

save_to_json(data_mesh, file_path=f'{opts.out_path}/data_mesh_pvc.json')

"""
python infer_wild_mesh_list.py --out_path myoutput/qualitation/mesh_pvcp_test_from_2ddet_rebuttal
"""