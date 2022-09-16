import torch
import os
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from nmg.utils import transforms as p3dt
import numpy as np
import pickle as pkl
import numpy as np
from ..utils.nmg_repr import rottrans2transf, smpl2nmg
from ..utils import config

home_dir = "/is/ps3/nsaini"



def get_amass_train_test(hparams,split_ratio=0.8):
    '''
    '''
    train_amass = amass(hparams)
    test_amass = amass(hparams)
    data_len = int(np.round(len(train_amass.npz_files)*split_ratio))
    train_amass.npz_files = train_amass.npz_files[:data_len]
    test_amass.npz_files = test_amass.npz_files[data_len:]

    return train_amass, test_amass


# osp.join(home_dir,"datasets/amass_babel/npz_files_fps_120_60.pkl")

class amass(Dataset):

    def __init__(self, hparams):
        super().__init__()

        datapath = hparams["datapath"]
        seq_begin = hparams["seq_begin"]
        input_seq_len = hparams["input_seq_len"]
        output_seq_len = hparams["output_seq_len"]
        mean_std_path = hparams["mean_std_path"]

        self.npz_files = pkl.load(open(datapath,"rb"))

        if (input_seq_len is None or output_seq_len is None):
            assert (input_seq_len is None and output_seq_len is None), "Both the input and output sequence length should be None, if either is None"
            self.seq_len = None
        else:
            assert input_seq_len <= output_seq_len, "output sequence length should be greater than or equal to the input sequence length"
            self.seq_len = np.max([input_seq_len,output_seq_len])
            
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        
        if seq_begin is not None:
            assert seq_begin >= 0, "sequence begin should be None or greater than zero"
        self.seq_begin = seq_begin
        self.totensor = transforms.ToTensor()
        if mean_std_path is not None:
            mean_std = np.load(mean_std_path)
            self.data_mean = torch.from_numpy(mean_std["mean"]).float()
            self.data_std = torch.from_numpy(mean_std["std"]).float()
        else:
            self.data_mean = 0.
            self.data_std = 1.
        

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index, begin=None, end=None):
        
        data = np.load(osp.join(config.dataset_dir, self.npz_files[index]))
        fps = data["mocap_framerate"]

        if fps == 120:
            trans = self.totensor(data["trans"][::2]).squeeze(0).float()
            poses = self.totensor(data["poses"][::2,:22*3]).squeeze(0).float()
        else:
            trans = self.totensor(data["trans"]).squeeze(0).float()
            poses = self.totensor(data["poses"][:,:22*3]).squeeze(0).float()

        if (begin is not None) and (end is not None):
            trans = trans[begin:end]
            poses = poses[begin:end]
            seq_start = begin
        elif self.seq_len is not None:
            if self.seq_begin is None:
                seq_start = np.random.randint(0,trans.shape[0]-self.seq_len) 
            else:
                seq_start = self.seq_begin
            trans = trans[seq_start:seq_start + self.seq_len]
            poses = poses[seq_start:seq_start + self.seq_len]
        else:
            seq_start = 0
        
        # translation starting from zero (x and y)
        poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))
        # trans[:,[0,2]] -= trans[0,[0,2]]
        
        fwd = poses_matrix[:,0,:3,2]
        fwd[:,2] = 0
        fwd /= torch.linalg.norm(fwd,dim=1).unsqueeze(1)
        right = torch.cross(torch.tensor([[0,0,1]]).float().repeat([fwd.shape[0],1]),fwd,dim=1)
        right /= torch.linalg.norm(right,dim=1).unsqueeze(1)
        up = torch.tensor([[0,0,1]]).float().repeat([fwd.shape[0],1])
        target_orient = torch.cat([right.unsqueeze(2),up.unsqueeze(2),fwd.unsqueeze(2)],dim=2)

        target_trans = trans[0].clone()
        target_trans[2] = 0

        new_orig_transf = rottrans2transf(target_orient[0].unsqueeze(0),target_trans.unsqueeze(0))
        seq_begin_root_transf = rottrans2transf(poses_matrix[:,0],trans)
        
        seq_transf_wrt_new_orig = torch.matmul(torch.inverse(new_orig_transf),seq_begin_root_transf)
        poses_matrix[:,0] = seq_transf_wrt_new_orig[:,:3,:3]
        trans = seq_transf_wrt_new_orig[:,:3,3]
        
        norm_poses = torch.cat([trans,p3dt.matrix_to_axis_angle(poses_matrix).reshape(trans.shape[0],-1)],dim=1)
        poses_nmg_repr = smpl2nmg(norm_poses).reshape(norm_poses.shape[0],22*9)
        poses_nmg_repr = (poses_nmg_repr-self.data_mean)/self.data_std

        return {"data":poses_nmg_repr[:self.input_seq_len], 
                "output_data": poses_nmg_repr[:self.output_seq_len],
                "seq_start": seq_start}



def get_mean_std(hparams):

    amass_dset = amass(hparams)

    from tqdm import tqdm

    data_sum = 0
    data_len = 0

    for idx in tqdm(range(len(amass_dset))):
        data = np.load(amass_dset.npz_files[idx])

        fps = data["mocap_framerate"]

        if fps == 120:
            trans = amass_dset.totensor(data["trans"][::2]).squeeze(0).float()
            poses = amass_dset.totensor(data["poses"][::2,:22*3]).squeeze(0).float()
        else:
            trans = amass_dset.totensor(data["trans"]).squeeze(0).float()
            poses = amass_dset.totensor(data["poses"][:,:22*3]).squeeze(0).float()
        # translation starting from zero
        trans[:,:2] -= trans[0,:2]

        # Root orient start from identity matrix
        poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))

        # convert from matrices to 6D representation
        poses_6d = p3dt.matrix_to_rotation_6d(poses_matrix).view(poses.shape[0],-1)
        
        trans_w_poses = torch.cat([trans,poses_6d],dim=1).float().view(-1)
        trans_w_poses = trans_w_poses.reshape(trans.shape[0],3+22*6)

        data_sum += torch.sum(trans_w_poses,dim=0).data.cpu().numpy()
        data_len += trans.shape[0]

    data_mean = data_sum/data_len

    data_var_sum = 0

    for idx in tqdm(range(len(amass_dset))):
        data = np.load(amass_dset.npz_files[idx])

        fps = data["mocap_framerate"]

        if fps == 120:
            trans = amass_dset.totensor(data["trans"][::2]).squeeze(0).float()
            poses = amass_dset.totensor(data["poses"][::2,:22*3]).squeeze(0).float()
        else:
            trans = amass_dset.totensor(data["trans"]).squeeze(0).float()
            poses = amass_dset.totensor(data["poses"][:,:22*3]).squeeze(0).float()
        # translation starting from zero
        trans[:,:2] -= trans[0,:2]

        # Root orient start from identity matrix
        poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))

        # convert from matrices to 6D representation
        poses_6d = p3dt.matrix_to_rotation_6d(poses_matrix).view(poses.shape[0],-1)
        
        trans_w_poses = torch.cat([trans,poses_6d],dim=1).float().view(-1)
        trans_w_poses = trans_w_poses.reshape(trans.shape[0],3+22*6)

        data_var_sum += np.sum((trans_w_poses.data.cpu().numpy()-data_mean)**2,axis=0)

    data_std = np.sqrt(data_var_sum/data_len)

    np.savez(osp.join(home_dir,"datasets/amass_babel/nmg_zero_trans_xy_frame_mean_std.npz"),mean=data_mean,std=data_std)
    
