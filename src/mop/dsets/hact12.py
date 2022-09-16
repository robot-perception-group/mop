import torch
import os
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from nmg.utils import transforms as p3dt
import numpy as np
import pickle as pkl
import numpy as np
from itertools import compress


def get_hact12_train_test(hparams,split_ratio=0.8,cats=None, seed=0):
    train_hact12 = hact12(hparams,cats)
    test_hact12 = hact12(hparams,cats)
    train_idcs = np.zeros(len(train_hact12))
    train_idcs[:int(len(train_hact12)*split_ratio)] = 1
    train_idcs = train_idcs.astype(np.bool)
    
    if seed is not None:
        np.random.seed(seed)
    
    np.random.shuffle(train_idcs)
    test_idcs = (1-train_idcs).astype(np.bool)

    train_hact12.poses = list(compress(train_hact12.poses,train_idcs))
    train_hact12.classes = list(compress(train_hact12.classes,train_idcs))

    test_hact12.poses = list(compress(test_hact12.poses,test_idcs))
    test_hact12.classes = list(compress(test_hact12.classes,test_idcs))

    return train_hact12, test_hact12


class hact12(Dataset):
    
    def __init__(self,hparams,cats=None):
        super().__init__()

        datapath = hparams["datapath"]
        seq_begin = hparams["seq_begin"]
        input_seq_len = hparams["input_seq_len"]
        output_seq_len = hparams["output_seq_len"]
        mean_std_path = hparams["mean_std_path"]

        data = pkl.load(open(datapath,"rb"))
        poses = []
        classes = []

        if (input_seq_len is None or output_seq_len is None):
            assert (input_seq_len is None and output_seq_len is None), "Both the input and output sequence length should be None, if either is None"
            self.seq_len = None
            poses = data["poses"]
            classes = data["y"]
        else:
            assert input_seq_len <= output_seq_len, "output sequence length should be greater than or equal to the input sequence length"
            self.seq_len = output_seq_len
            # filer based on the length
            for id in range(len(data["poses"])):
                if data["poses"][id].shape[0] > self.seq_len:
                    poses.append(data["poses"][id])
                    classes.append(data["y"][id])
            
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

        # filter based on the category
        if cats is not None:
            assert type(cats) is list, "cats should be a list"
            self.poses = [poses[id] for id in range(len(poses)) if classes[id] in cats]
            self.classes = [classes[id] for id in range(len(classes)) if classes[id] in cats]

    
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        
        if self.seq_len is not None:
            if self.seq_begin is None:
                seq_start = np.random.randint(0,self.poses[index].shape[0]-self.seq_len) 
            else:
                seq_start = self.seq_begin
            trans = torch.zeros(self.seq_len,3).float()
            poses = torch.from_numpy(self.poses[index][seq_start:seq_start + self.seq_len,:22*3]).float()
        else:
            seq_start = 0
            poses = torch.from_numpy(self.poses[index])
            trans = torch.zeros(poses.shape[0],3).float()

        
        # translation starting from zero (x and y)
        poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))
        trans[:,:2] -= trans[0,:2]
        
        # convert from matrices to 6D representation
        poses_6d = (p3dt.matrix_to_rotation_6d(poses_matrix).view(poses.shape[0],-1) - self.data_mean[:22*6])/self.data_std[:22*6]
        
        trans_w_poses = torch.cat([trans,poses_6d],dim=1).float()
        
        return {"data":trans_w_poses[:self.input_seq_len], 
                "output_data": trans_w_poses[:self.output_seq_len],
                "seq_start": seq_start, 
                "frame_anns": 0,
                "contacts": 0}


def load_pose_and_shape(data_dir, subject_no='all'):
    annotations = {}
    filenames = sorted(os.listdir('%s/pose/' % data_dir))
    for idx, filename in enumerate(filenames):
        if subject_no not in filename and subject_no != 'all':
            continue
        else:
            print('processed %s %i / %i' % (filename, idx, len(filenames)))
            # read smpl shape parameters as dict
            shape_params = {}
            with open('%s/pose/%s/shape_smpl.txt' % (data_dir, filename), 'r') as f:
                for line in f.readlines():
                    tmp = line.split(' ')  # frame_idx
                    smpl_param = np.asarray([float(i) for i in tmp[1:]])
                    # [beta, translation, theta]
                    smpl_param = np.concatenate([smpl_param[3:13], smpl_param[0:3], smpl_param[13:85]], axis=0)
                    shape_params[tmp[0]] = smpl_param

            # read 3D joint pose
            with open('%s/pose/%s/pose.txt' % (data_dir, filename), 'r') as f:
                for line in f.readlines():
                    tmp = line.split(' ')  # frame_idx
                    pose = np.asarray([float(i) for i in tmp[1:]]).reshape([-1, 3])
                    annotations['%s-%s' % (filename, tmp[0])] = (pose, shape_params[tmp[0]])
    return annotations
