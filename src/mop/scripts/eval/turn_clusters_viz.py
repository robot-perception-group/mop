import sys, os, pdb
from os.path import join as ospj
import json
from collections import *

import numpy as np
import pandas as pd
from pandas.core import common
from pandas.core.common import flatten

import pprint
from sklearn.utils.validation import has_fit_parameter
pp = pprint.PrettyPrinter()

from tqdm import tqdm
import pickle as pkl

# Amass humor dataset files
npz_list = pkl.load(open("/home/nsaini/Datasets/AMASS_humor_processed/amass_humor_all.pkl","rb"))
data_root_dir = npz_list[0].split("/")[0]
suffixes = ["_".join(x.split("_")[-4:]) for x in npz_list]
npz_list = ["/".join("_".join(x.split("_")[:-4]).split("/")[1:]) + ".npz" for x in npz_list]
seq2suffix = {npz_list[i]:suffixes[i] for i in range(len(npz_list))}


d_folder = '/home/nsaini/Datasets/babel_v1-0_release/babel_v1.0_release' # Data folder
l_babel_dense_files = ['train', 'val']  
l_babel_extra_files = ['extra_train', 'extra_val']

# BABEL Dataset 
babel = {}
for file in l_babel_dense_files:
    babel[file] = json.load(open(ospj(d_folder, file+'.json')))
    
# for file in l_babel_extra_files:
#     babel[file] = json.load(open(ospj(d_folder, file+'.json')))


# %%
def get_cats(ann, file):
    # Get sequence labels and frame labels if they exist
    seq_l, frame_l = [], []
    if 'extra' not in file:
        if ann['seq_ann'] is not None:
            seq_l = flatten([seg['act_cat'] for seg in ann['seq_ann']['labels']])
        if ann['frame_ann'] is not None:
            frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = flatten([seg['act_cat'] for seq_ann in ann['seq_anns'] for seg in seq_ann['labels']])
        if ann['frame_anns'] is not None:            
            frame_l = flatten([seg['act_cat'] for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']])
            
    return list(seq_l), list(frame_l)

# %%
seq_list = []

for spl in tqdm(babel):
    for sid in tqdm(babel[spl]):
            if babel[spl][sid]["frame_ann"] is None:
                if 0.8*babel[spl][sid]["dur"] >= 1:
                    seq_list.append({"path":"/".join(babel[spl][sid]["feat_p"].split("/")[1:]),
                                        "begin":0,
                                        "end":babel[spl][sid]["dur"]*0.8,
                                        "orig_totat_time": babel[spl][sid]["dur"],
                                        "cat":list(flatten([x["act_cat"] for x in babel[spl][sid]["seq_ann"]["labels"]]))})
            else:
                for label in babel[spl][sid]["frame_ann"]["labels"]:
                    if (label["start_t"] > 0.1*babel[spl][sid]["dur"]) and                             (label["end_t"] - label["start_t"]) >= 1 and                                 (label["end_t"] < 0.9*babel[spl][sid]["dur"]):
                        seq_list.append({"path":"/".join(babel[spl][sid]["feat_p"].split("/")[1:]),
                                        "begin":label["start_t"] - 0.1*babel[spl][sid]["dur"],
                                        "end":label["end_t"] - 0.1*babel[spl][sid]["dur"],
                                        "orig_totat_time": babel[spl][sid]["dur"],
                                        "cat":label["act_cat"]})

common_seq = []
for i in tqdm(range(len(seq_list))):
    if seq_list[i]["path"] in npz_list:
        common_seq.append(seq_list[i])
        common_seq[-1]["path"] = ospj(data_root_dir,".".join(common_seq[-1]["path"].split(".")[:-1]) + "_" + seq2suffix[common_seq[-1]["path"]])
    

# %% Select sequences
cats, cat_count = np.unique(list(flatten([x["cat"] for x in common_seq if len(x["cat"])==1])),return_counts=True)
cats = cats[cat_count>100]
cat_count = cat_count[cat_count>100]

selected_sequences = ["turn"]
num_selected_sequences = []
sorted_common_seq = []
for cat in tqdm(selected_sequences):
    sequences = sorted([x for x in common_seq if ((cat in x["cat"]) and (len(x["cat"])==1))],key= lambda x: x["path"],reverse=True)
    sorted_common_seq += sequences
    num_selected_sequences += [len(sequences)]


# %% Load trained model
from nmg.utils import nmg_repr
from nmg.models import nmg
import torch
import yaml
ckpt_path = "/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/conv_nmg_dvae_NoKLAnnealing_2021_12_17/v0_rec_pose_weight=1,codebook_size=500,min_gumbel_softmax_temp=0.5,gumbel_temp_anneal_rate=0.001/checkpoints/epoch=2589-step=821029.ckpt"
hparams = yaml.load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml"),Loader=yaml.FullLoader)
model = nmg.nmg.load_from_checkpoint(ckpt_path,hparams=hparams)
model.eval()

######## assign proper gumbel softmax temp ###############
model.gumbel_softmax_temp = model.hparams["min_gumbel_softmax_temp"]
########################

from nmg.dsets import amass_humor

output_seq_len = model.hparams["output_seq_len"]

# load dataset
if model.hparams["data"].lower() == "amass_humor":
    train_dset, val_dset = amass_humor.get_amass_humor_train_test(model.hparams)
else:
    import ipdb;ipdb.set_trace()


dset = amass_humor.amass_humor({"datapath":model.hparams["datapath"],
                                    "seq_begin":0,
                                    "input_seq_len":None,
                                    "output_seq_len":None,
                                    "mean_std_path":model.hparams["mean_std_path"]})

dset.npz_files = [x["path"] for x in sorted_common_seq]

z_all = []
inputs = []
encoder_outputs = []

for id in tqdm(range(0,len(dset)-100,100)):
    dsample = []
    for i in range(id,id+100):
        begin = np.random.randint(int(np.ceil(sorted_common_seq[i]["begin"]*30)), int(np.floor(sorted_common_seq[i]["end"]*30))-26)
        if begin < 0:
            begin=0
        end = begin + 25
        dsample.append(dset.__getitem__(i,begin=begin,end=end)["data"])
    dsample = torch.stack(dsample)
    encoder_output, decoder_output, z = model(dsample)
    z_all.append(z)
    inputs.append(dsample)
    encoder_outputs.append(encoder_output)

id += 100
dsample = []
for i in range(id,len(sorted_common_seq)):
    begin = np.random.randint(int(np.ceil(sorted_common_seq[i]["begin"]*30)), int(np.floor(sorted_common_seq[i]["end"]*30))-26)
    if begin < 0:
        begin=0
    end = begin + 25
    dsample.append(dset.__getitem__(i,begin=begin,end=end)["data"])
dsample = torch.stack(dsample)
encoder_output, decoder_output, z = model(dsample)
z_all.append(z)
inputs.append(dsample)
encoder_outputs.append(encoder_output)
    

inputs = torch.cat(inputs).detach().cpu().numpy()
encoder_outputs = torch.cat(encoder_outputs).detach().cpu().numpy()
z_all = torch.cat(z_all)
z_all = z_all.reshape(z_all.shape[0],-1).detach().cpu().numpy()

seq_lens = np.cumsum(num_selected_sequences)
seq_indices = {selected_sequences[i]:range(seq_lens[i-1],seq_lens[i]) for i in range(1,len(seq_lens))}
seq_indices[selected_sequences[0]] = range(seq_lens[0])

# %% Clustering
from sklearn.cluster import KMeans

n_clusters = 2

z_km = KMeans(n_clusters=n_clusters).fit(z_all)

import umap
z_umap = umap.UMAP().fit_transform(z_all)

# plotting
# import matplotlib.pyplot as plt
# plt.scatter(z_umap[z_km.labels_==0,0],z_umap[z_km.labels_==0,1])
# plt.scatter(z_umap[z_km.labels_==1,0],z_umap[z_km.labels_==1,1])
# plt.show()

# %% for blender

turn_samples1 = torch.from_numpy(inputs[z_km.labels_==0][np.random.randint(0,sum(z_km.labels_==0),5)]).float()
turn_samples2 = torch.from_numpy(inputs[z_km.labels_==1][np.random.randint(0,sum(z_km.labels_==1),5)]).float()

turn_samples1_smpl = nmg_repr.nmg2smpl((turn_samples1*dset.data_std+dset.data_mean).reshape(-1,22,9)).reshape(-1,turn_samples1.shape[1],22*3+3)
turn_samples2_smpl = nmg_repr.nmg2smpl((turn_samples2*dset.data_std+dset.data_mean).reshape(-1,22,9)).reshape(-1,turn_samples2.shape[1],22*3+3)

turn_samples_smpl = torch.cat([turn_samples1_smpl,turn_samples2_smpl])

os.makedirs(".".join(ckpt_path.split(".")[:-1]),exist_ok=True)
np.savez(".".join(ckpt_path.split(".")[:-1])+"/"+"_".join(selected_sequences)+"_turn_cluster_viz",
                    root_orient=turn_samples_smpl[:,:,3:6].detach().cpu().numpy(),
                    pose_body=turn_samples_smpl[:,:,6:].detach().cpu().numpy(),
                    trans=turn_samples_smpl[:,:,:3].detach().cpu().numpy())

