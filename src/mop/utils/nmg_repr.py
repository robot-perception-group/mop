
from nmg.utils import transforms as p3dt
from human_body_prior.body_model.body_model import BodyModel
import os
import os.path as osp
import torch
from . import config

bm = BodyModel(osp.join(config.dataset_dir,"smpl_models/smplh/neutral/model.npz"))
bm.eval()

def rottrans2transf(rotmat,trans):
    batch_size = rotmat.shape[0]
    assert rotmat.shape[0]==trans.shape[0], "rotmat and trans should have same batch size"
    return torch.cat([torch.cat([rotmat,trans.unsqueeze(2)],dim=2),torch.tensor([0,0,0,1]).unsqueeze(0).unsqueeze(0).float().repeat(batch_size,1,1).to(rotmat.device)],dim=1)

def smpl2nmg(poses):
    joints = bm.forward(root_orient=poses[:,3:6],pose_body=poses[:,6:]).Jtr
    joint_pos_wrt_root = joints[:,1:] - joints[:,0:1]
    transfs = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(poses.shape[0],22,1,1).to(poses.device)
    pose_angles = poses[:,3:].view(poses.shape[0],-1,3)

    transfs[:,0] = rottrans2transf(p3dt.axis_angle_to_matrix(pose_angles[:,0]),joints[:,0])
    for j in range(1,22):
        transfs[:,j] = rottrans2transf(torch.matmul(transfs[:,bm.kintree_table[0,j],:3,:3],p3dt.axis_angle_to_matrix(pose_angles[:,j])),joint_pos_wrt_root[:,j])
    
    transfs[:,:,:3,3] += poses[:,:3].unsqueeze(1)

    nmg_transfs = torch.zeros(poses.shape[0],22,9).float().to(poses.device)
    nmg_transfs[:,:,:6] = p3dt.matrix_to_rotation_6d(transfs[:,:,:3,:3])
    nmg_transfs[:,:,6:] = transfs[:,:,:3,3]

    return nmg_transfs


def nmg2smpl(nmg_transfs):
    transfs = torch.zeros(nmg_transfs.shape[0],nmg_transfs.shape[1],4,4).float().to(nmg_transfs.device)
    transfs[:,:,:3,:3] = p3dt.rotation_6d_to_matrix(nmg_transfs[:,:,:6])
    transfs[:,:,:3,3] = nmg_transfs[:,:,6:]
    poses_angles = torch.zeros(transfs.shape[0],transfs.shape[1],3).float().to(nmg_transfs.device)
    
    for j in range(21,0,-1):
        poses_angles[:,j] = p3dt.matrix_to_axis_angle(torch.matmul(torch.inverse(transfs[:,bm.kintree_table[0,j],:3,:3]),transfs[:,j,:3,:3]))
    poses_angles[:,0] = p3dt.matrix_to_axis_angle(transfs[:,0,:3,:3])

    joints = bm.forward(root_orient=poses_angles[:,0],pose_body=poses_angles.view(poses_angles.shape[0],22*3)[:,3:]).Jtr

    trans = transfs[:,0,:3,3] - joints[:,0]

    return torch.cat([trans,poses_angles.reshape(trans.shape[0],22*3)],dim=1)




    