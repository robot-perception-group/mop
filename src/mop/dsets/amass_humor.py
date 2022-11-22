import torch
import os
import os.path as osp
from torch.functional import norm
from torch.utils.data import Dataset
from torchvision import transforms
from mop.utils import transforms as p3dt
import numpy as np
import pickle as pkl
import numpy as np
from tqdm import tqdm
from ..utils.nmg_repr import rottrans2transf, smpl2nmg
from ..utils import config
from human_body_prior.body_model.body_model import BodyModel

home_dir = "/is/ps3/nsaini"

bm = BodyModel(osp.join(config.dataset_dir,"smpl_models/smplh/neutral/model.npz"))
bm.eval()



def get_amass_humor_train_test(hparams,split_ratio=0.8):
    '''
    '''
    train_amass_humor = amass_humor(hparams)
    test_amass_humor = amass_humor(hparams)
    data_len = int(np.round(len(train_amass_humor.npz_files)*split_ratio))
    train_amass_humor.npz_files = train_amass_humor.npz_files[:data_len]
    test_amass_humor.npz_files = test_amass_humor.npz_files[data_len:]

    return train_amass_humor, test_amass_humor


# osp.join(home_dir,"datasets/amass_babel/npz_files_fps_120_60.pkl")

class amass_humor(Dataset):

    def __init__(self, hparams=None):
        super().__init__()

        if hparams == None:
            hparams = {"datapath":"/home/nsaini/Datasets/AMASS_humor_processed/amass_humor_all.pkl",
                        "seq_begin": None,
                        "input_seq_len":25,
                        "output_seq_len":25,
                        "mean_std_path":"/home/nsaini/Datasets/AMASS_humor_processed/amass_humor_all_mean_std.npz"}

        datapath = hparams["datapath"]
        seq_begin = hparams["seq_begin"]
        input_seq_len = hparams["input_seq_len"]
        output_seq_len = hparams["output_seq_len"]
        mean_std_path = hparams["mean_std_path"]

        self.npz_files = sorted(pkl.load(open(datapath,"rb")))
        self.device="cpu"

        # loading the npz files in memory
        self.load_from_npz_files()

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
        
        if mean_std_path is not None:
            mean_std = np.load(mean_std_path)
            self.data_mean = torch.from_numpy(mean_std["mean"]).float().to(self.device)
            self.data_std = torch.from_numpy(mean_std["std"]).float().to(self.device)
        else:
            self.data_mean = 0.
            self.data_std = 1.
        
    def load_from_npz_files(self):
        print("##########loading the npz files in memory############")
        self.data_dict = {}
        for x in self.npz_files:
            data = np.load(osp.join(config.dataset_dir, x))
            self.data_dict[x] = {"trans":torch.from_numpy(data["trans"]).to(self.device), 
                                "root_orient":torch.from_numpy(data["root_orient"]).to(self.device),
                                "pose_body":torch.from_numpy(data["pose_body"]).to(self.device),
                                "contacts":torch.from_numpy(data["contacts"]).to(self.device)}

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index, begin=None, end=None):
        

        
        data = self.data_dict[self.npz_files[index]]
        
        trans = data["trans"].squeeze(0).float()
        root_orient = data["root_orient"].squeeze(0).float()
        pose_body = data["pose_body"][:,:22*3].squeeze(0).float()
        poses = torch.cat([root_orient,pose_body],dim=1)

        if (begin is not None) and (end is not None):
            trans = trans[begin:end]
            poses = poses[begin:end]
            contacts = data["contacts"][begin:end]
            seq_start = begin
        elif self.seq_len is not None:
            if self.seq_begin is None:
                seq_start = np.random.randint(0,trans.shape[0]-self.seq_len) 
            else:
                seq_start = self.seq_begin
            trans = trans[seq_start:seq_start + self.seq_len]
            poses = poses[seq_start:seq_start + self.seq_len]
            contacts = data["contacts"][seq_start:seq_start + self.seq_len]
            end = self.input_seq_len
        else:
            seq_start = 0
        
        trans[:,:2] = trans[:,:2] - trans[0,:2]

        norm_poses = get_norm_poses(poses,trans)
        
        poses_nmg_repr = smpl2nmg(norm_poses).reshape(norm_poses.shape[0],22*9)
        poses_nmg_repr = (poses_nmg_repr-self.data_mean)/self.data_std

        return {"data":poses_nmg_repr[:end], 
                "output_data": poses_nmg_repr[:self.output_seq_len],
                "contacts": contacts,
                "seq_start": seq_start}


def get_norm_poses(poses, trans):

    # translation starting from zero (x and y)
    poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))
    # trans[:,[0,2]] -= trans[0,[0,2]]
    # import ipdb;ipdb.set_trace()
    fwd = poses_matrix[0,0,:3,2].clone()
    fwd[2] = 0
    fwd /= torch.linalg.norm(fwd)
    if fwd[0] > 0:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))
    else:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,-torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))
    
    tfmd_orient = torch.matmul(tfm,poses_matrix[:,0])
    tfmd_trans = torch.matmul(tfm,trans.unsqueeze(2)).squeeze(2)
    
    poses_matrix[:,0] = tfmd_orient
    
    norm_poses = torch.cat([tfmd_trans,p3dt.matrix_to_axis_angle(poses_matrix).reshape(tfmd_trans.shape[0],-1)],dim=1)

    return norm_poses