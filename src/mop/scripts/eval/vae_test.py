from nmg.utils import nmg_repr
from nmg.models import nmg
import torch
import yaml
import os
import numpy as np

ckpt_path = "/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/conv_nmg_vae_211221-193320/nmg_logs/conv_nmg_vae_2021_12_15/train_nmg_fea95_00004_4_kl_annealing_cycle_epochs=10,rec_pose_weight=5_2021-12-16_03-19-39/checkpoints/epoch=869-step=275789.ckpt"
hparams = yaml.load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml"),Loader=yaml.FullLoader)
model = nmg.nmg.load_from_checkpoint(ckpt_path,hparams=hparams)
model.eval()

######## assign proper gumbel softmax temp ###############
model.gumbel_softmax_temp = model.hparams["min_gumbel_softmax_temp"]
########################

from nmg.dsets import amass_humor

# load dataset
if model.hparams["data"].lower() == "amass_humor":
    train_dset, val_dset = amass_humor.get_amass_humor_train_test(model.hparams)

# random samples
samples = model.decode(2*torch.randn(10,1024).float())

samples_smpl = nmg_repr.nmg2smpl((samples*train_dset.data_std+train_dset.data_mean).reshape(-1,22,9)).reshape(-1,samples.shape[1],22*3+3)

os.makedirs(".".join(ckpt_path.split(".")[:-1]),exist_ok=True)
np.savez(".".join(ckpt_path.split(".")[:-1])+"/"+"rand_samples",
                    root_orient=samples_smpl[:,:,3:6].detach().cpu().numpy(),
                    pose_body=samples_smpl[:,:,6:].detach().cpu().numpy(),
                    trans=samples_smpl[:,:,:3].detach().cpu().numpy())