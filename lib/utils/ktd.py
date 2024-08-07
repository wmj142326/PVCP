import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

ANCESTOR_INDEX = [
    [],
    [0], 
    [0], 
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 1, 4],
    [0, 2, 5],
    [0, 3, 6],
    [0, 1, 4, 7],
    [0, 2, 5, 8],
    [0, 3, 6, 9], 
    [0, 3, 6, 9], 
    [0, 3, 6, 9],
    [0, 3, 6, 9, 12],
    [0, 3, 6, 9, 13],
    [0, 3, 6, 9, 14],
    [0, 3, 6, 9, 13, 16],
    [0, 3, 6, 9, 14, 17],
    [0, 3, 6, 9, 13, 16, 18],
    [0, 3, 6, 9, 14, 17, 19],
    [0, 3, 6, 9, 13, 16, 18, 20],
    [0, 3, 6, 9, 14, 17, 19, 21]
]

class KTD(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=512, **kwargs):
        super(KTD, self).__init__()

        self.feat_dim = feat_dim
        npose_per_joint = 6

        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        
        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(ANCESTOR_INDEX):
            regressor = nn.Linear(hidden_dim + npose_per_joint * len(ancestor_idx), npose_per_joint)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

    def forward(self, x, **kwargs):
        nt = x.shape[0]

        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        pose = []
        for ancestor_idx, reg in zip(ANCESTOR_INDEX, self.joint_regs):
            ances = torch.cat([x] + [pose[i] for i in ancestor_idx], dim=1)
            pose.append(reg(ances))

        pred_pose = torch.cat(pose, dim=1)

        return pred_pose