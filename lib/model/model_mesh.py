import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class SMPLRegressor(nn.Module):
    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        self.head_shape = nn.Linear(hidden_dim, 10)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)
        self.smpl = SMPL(
            args.data_root,
            batch_size=64,
            create_transl=False,
        )
        mean_params = np.load(self.smpl.smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None, n_iter=3):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)  
        feat_pose = feat.reshape(NT, -1)     # (N*T, J*C)

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)  # (N*T, J*C)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)    # (NT, C)

        feat_shape = feat.permute(0,2,1)     # (N, T, J*C) -> (N, J*C, T)
        feat_shape = self.pool2(feat_shape).reshape(N, -1)          # (N, J*C)

        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)     # (N, C)

        pred_pose = self.init_pose.expand(NT, -1)   # (NT, C)
        pred_shape = self.init_shape.expand(N, -1)  # (N, C)

        # for i in range(n_iter):
        #     pred_pose = self.head_pose(feat_pose) + pred_pose
        #     pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_pose = self.head_pose(feat_pose) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, N, -1).permute(1, 0, 2).reshape(NT, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices*1000.0
        assert self.J_regressor is not None
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{
            'theta'  : torch.cat([pose, pred_shape], dim=1),    # (N*T, 72+10)
            'verts'  : pred_vertices,                           # (N*T, 6890, 3)
            'kp_3d'  : pred_joints,                             # (N*T, 17, 3)
        }]
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DecodePVCP(nn.Module):
    def __init__(self, in_planes):
        super(DecodePVCP, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        
        x = x.permute(0,3,1,2)
        residual = x
        
        x = self.ca(x) * x
        x = self.sa(x) * x
        
        x += residual
        x = self.relu(x)
        x = x.permute(0,2,3,1)
        
        return x
    
class FeatureExtractor(nn.Module):
    def __init__(self, num_joint=17):
        super(FeatureExtractor, self).__init__()
        self.num_joint = num_joint
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # 特征金字塔网络，使用3x3的卷积核进行多尺度特征融合
        self.pyramid = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.fc = nn.Linear(512*8*8, 512)

    def forward(self, x):
        N, T, _,C, H, W = x.size()
        x = x.view(-1, C, H, W)
        features = self.resnet_features(x)
        features = features.view(N, T, -1, features.size(2), features.size(3))
        pyramid_features = self.pyramid(features.mean(dim=1))
        pyramid_features = pyramid_features.unsqueeze(1).expand(N, T, -1, -1, -1)
        out = pyramid_features.reshape(N,T,-1)
        out = self.fc(out)
        out = out.unsqueeze(2).expand(-1, -1, self.num_joint, -1)
        return out

            
class MeshRegressor(nn.Module):
    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5, att_fuse=False):
        super(MeshRegressor, self).__init__()
        self.args = args
        self.dim_rep = dim_rep
        self.backbone = backbone
        # freeze_pretrain
        if args.freeze_pretrain:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.feat_J = num_joints
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)
        
        
        # ----------image_feature-----------
        if args.image_feature:
            self.get_img_feature = FeatureExtractor()
            self.fc = nn.Linear(dim_rep*2, dim_rep)
        
            # ----------fuse_feature----------        
            self.att_fuse = att_fuse  
            if self.att_fuse:
                self.ts_attn = nn.Linear(dim_rep*2, dim_rep*2)
        
        # ----------decoder----------
        if args.decoder:
            self.meshdecode = DecodePVCP(dim_rep)

    def forward(self, x_2d, x_sensor, init_pose=None, init_shape=None):
        '''
            Input: (N x T x 17 x 3) 
        '''
        N, T, J, C = x_2d.shape
        feat = self.backbone.get_representation(x_2d)        # (N, T, J, 512)
        
        # ----------image_feature-----------
        if self.args.image_feature:
            feat_img = self.get_img_feature(x_sensor['img_feature'])        
        
            # ----------fuse_feature----------        
            # if self.att_fuse:
            #     feat_2d=feat.reshape(N*T, J, self.dim_rep)
            #     feat_img = feat_img.reshape(N*T,J, self.dim_rep)
            #     alpha = torch.cat([feat_2d, feat_img], dim=-1)
            #     BF, J = alpha.shape[:2]
            #     alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
            #     alpha = alpha.softmax(dim=-1)
            #     feat = feat_2d * alpha[:,:,:,0] + feat_img * alpha[:,:,:,1]
            #     feat = feat.reshape(N,T,J,-1)
            # else:           
            #     feat = torch.cat((feat, feat_img), dim=-1)  # cat
            #     feat = self.fc(feat)
            
            feat = torch.cat((feat, feat_img), dim=-1)  # cat
            feat = self.fc(feat)
        
        # ----------decoder----------
        if self.args.decoder:
            feat = self.meshdecode(feat)    
    
              
        feat = feat.reshape([N, T, self.feat_J, -1])      # (N, T, J, C)
        smpl_output = self.head(feat)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        
        return smpl_output