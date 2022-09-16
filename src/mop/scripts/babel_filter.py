# %% Imports
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



# %% Load amass humor
npz_list = pkl.load(open("/home/nsaini/Datasets/AMASS_humor_processed/amass_humor_all.pkl","rb"))
data_root_dir = npz_list[0].split("/")[0]
suffixes = ["_".join(x.split("_")[-4:]) for x in npz_list]
npz_list = ["/".join("_".join(x.split("_")[:-4]).split("/")[1:]) + ".npz" for x in npz_list]
seq2suffix = {npz_list[i]:suffixes[i] for i in range(len(npz_list))}


# %% Load Babel

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
                    if (label["start_t"] > 0.1*babel[spl][sid]["dur"]) and \
                            (label["end_t"] - label["start_t"]) >= 1 and \
                                (label["end_t"] < 0.9*babel[spl][sid]["dur"]):
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
    

cats, cat_count = np.unique(list(flatten([x["cat"] for x in common_seq if len(x["cat"])==1])),return_counts=True)
cats = cats[cat_count>100]
cat_count = cat_count[cat_count>100]

selected_sequences = ["run","sit","stand","jump","jog","kick","turn"]
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
ckpt_path = "/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/2021-11-06_conv_nmg/train_nmg_52d4c_00003_3_codebook_size=1000,gumbel_temp_anneal_rate=0.005,min_gumbel_softmax_temp=0.5_2021-11-06_20-42-45/checkpoints/epoch=1039-step=329679 copy.ckpt"
hparams = yaml.load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml"),Loader=yaml.FullLoader)
model = nmg.nmg.load_from_checkpoint(ckpt_path,hparams=hparams)
model.eval()

from nmg.dsets import amass_humor

output_seq_len = model.hparams["output_seq_len"]

if model.hparams["data"].lower() == "amass_humor":
    train_dset, val_dset = amass_humor.get_amass_humor_train_test(model.hparams)
else:
    train_dset, val_dset = amass.get_amass_train_test(model.hparams)

# "val_recon" for validation samples reconstruction
# "train_recon" for training samples recontruction
# "rand_sample" for random samples from the latent space
for mode in ["val_recon","rand_sample","train_recon"]:

    if mode.lower() == "rand_sample":
        M = 10*model.hparams.latent_len
        np_one_hot = np.zeros((M,model.hparams.codebook_size))
        np_one_hot[range(M),np.random.choice(model.hparams.codebook_size,M)] = 1
        np_one_hot = np.reshape(np_one_hot,[-1,model.hparams.latent_len,model.hparams.codebook_size])
        one_hot = torch.from_numpy(np_one_hot).to(model.device).float()
        z = torch.einsum('b n c, c d -> b n d',one_hot,model.embeddings.weight)
        queries = model.posenc_dec_in(torch.zeros(z.shape[0], output_seq_len,z.shape[2])).to(z.device)
        decoder_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=z,tgt_mask=model.dec_mask))
        file_name_string = "latent_samples"
    else:
        if mode.lower() == "val_recon":

            inputs = [val_dset[np.random.randint(0,len(val_dset))] for _ in range(5)]
            file_name_string = "val_reconstruction"
        else:

            inputs = [train_dset[np.random.randint(0,len(train_dset))] for _ in range(5)]
            file_name_string = "train_reconstruction"

        dsample = torch.stack([x["data"] for x in inputs])
        encoder_output, decoder_output, z = model(dsample)
        decoder_output = torch.cat([dsample,decoder_output])




    batch_size = decoder_output.shape[0]
    if model.hparams.use_contacts:
        decoder_output_conts = decoder_output[:,:,3:].reshape(batch_size,output_seq_len,22,7)[:,:,:,6]
        decoder_output_pose = decoder_output[:,:,3:].reshape(batch_size,output_seq_len,22,7)[:,:,:,:6].reshape(batch_size,-1,22*6)
    else:
        decoder_output_smpl_repr = nmg_repr.nmg2smpl((decoder_output*train_dset.data_std + train_dset.data_mean).reshape(-1,22,9)).reshape(decoder_output.shape[0],decoder_output.shape[1],22*3+3)
        decoder_output_conts = None
        
    pose = decoder_output_smpl_repr[:,:,3:]
    trans = decoder_output_smpl_repr[:,:,:3]

    if model.hparams.use_contacts:
        np.savez(".".join(ckpt_path.split(".")[:-1])+file_name_string,
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    contacts=torch.sigmoid(decoder_output_conts).detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())
    else:
        np.savez(".".join(ckpt_path.split(".")[:-1])+file_name_string,
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())

# 

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

# from sklearn.manifold import TSNE
# z_tsne = TSNE(2).fit_transform(z_all)

import umap
z_umap = umap.UMAP().fit_transform(z_all)

clrs = []
for seq in sorted_common_seq:
    if seq["cat"][0] == "jog":
        clrs.append("tab:blue")
    elif seq["cat"][0] == "kick":
        clrs.append("tab:orange")
    elif seq["cat"][0] == "jump":
        clrs.append("tab:green")
    elif seq["cat"][0] == "walk":
        clrs.append("tab:red")
    elif seq["cat"][0] == "hand movements":
        clrs.append("tab:purple")
    elif seq["cat"][0] == "sit":
        clrs.append("tab:brown")
    elif seq["cat"][0] == "stand":
        clrs.append("tab:pink")
    elif seq["cat"][0] == "run":
        clrs.append("tab:gray")
    elif seq["cat"][0] == "throw":
        clrs.append("tab:olive")
    elif seq["cat"][0] == "turn":
        clrs.append("tab:cyan")
    else:
        clrs.append("k")

dist = np.zeros([z_all.shape[0],z_all.shape[0]])
for i in tqdm(range(z_all.shape[0])):
    dist[i,:] = np.sqrt(np.sum(((z_all-z_all[i:i+1,:])**2),axis=1))

import matplotlib.pyplot as plt
plt.figure(1)
plt.imshow(dist)
plt.xticks(ticks=np.cumsum(num_selected_sequences),labels=selected_sequences)
plt.yticks(ticks=np.cumsum(num_selected_sequences),labels=selected_sequences)
plt.savefig(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"confusion_mat.pdf")

plt.figure(2)
plt.scatter(z_umap[:seq_lens[0],0],z_umap[:seq_lens[0],1],c=clrs[:seq_lens[0]])
for i in range(seq_lens.shape[0]-1):
    plt.scatter(z_umap[seq_lens[i]:seq_lens[i+1],0],z_umap[seq_lens[i]:seq_lens[i+1],1],c=clrs[seq_lens[i]:seq_lens[i+1]])
plt.scatter(z_umap[seq_lens[-1]:,0],z_umap[seq_lens[-1]:,1],c=clrs[seq_lens[-1]:])
plt.legend(selected_sequences)
plt.savefig(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_space_viz.pdf")

import torch.nn.functional as F
fig,ax = plt.subplots(len(selected_sequences))
for i,seq in enumerate(selected_sequences):
    ax[i].imshow(F.gumbel_softmax(torch.from_numpy(encoder_outputs[seq_indices[seq],:,:]),
                    tau=model.hparams.min_gumbel_softmax_temp,dim=2).detach().cpu().numpy().mean(axis=0))
plt.savefig(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_activations.pdf",dpi=200)

# %% plot samples
# import random
# i=0
# np.savez(selected_sequences[i]+"_samples_babel.npz",root_orient=samples[random.sample(range(seq_lens[i]),15),:,3:6].detach().cpu().numpy(),
#                 pose_body=samples[random.sample(range(seq_lens[i]),15),:,6:],
#                 trans=samples[random.sample(range(seq_lens[i]),15),:,:3])
# for i in range(seq_lens.shape[0]-1):
#     np.savez(selected_sequences[i+1]+"_samples_babel.npz",root_orient=samples[random.sample(range(seq_lens[i],seq_lens[i+1]),15),:,3:6].detach().cpu().numpy(),
#                 pose_body=samples[random.sample(range(seq_lens[i],seq_lens[i+1]),15),:,6:],
#                 trans=samples[random.sample(range(seq_lens[i],seq_lens[i+1]),15),:,:3])





# %% Latent means
import torch.nn.functional as F
latent_class_means = torch.stack([torch.from_numpy(np.mean(encoder_outputs[seq_indices[x]],axis=0)) for x in selected_sequences])
soft_one_hot = F.softmax(latent_class_means,dim=2)
latent_class_means_z = torch.einsum('b n c, c d -> b n d',soft_one_hot,model.embeddings.weight)
queries = model.posenc_dec_in(torch.zeros(latent_class_means.shape[0], model.hparams.output_seq_len,latent_class_means_z.shape[2])).to(latent_class_means_z.device)
latent_class_means_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_means_z,tgt_mask=model.dec_mask))
latent_class_means_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_means_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_means_output.shape[0],latent_class_means_output.shape[1],22*3+3)

pose = latent_class_means_output_smpl_repr[:,:,3:]
trans = latent_class_means_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_means",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())

#%%
latent_class_interps = []
for i in range(latent_class_means.shape[0]-1):
    latent_class_interps += [latent_class_means[i],
                            latent_class_means[i]*0.75+latent_class_means[i+1]*0.25,
                            latent_class_means[i]*0.5+latent_class_means[i+1]*0.5,
                            latent_class_means[i]*0.25+latent_class_means[i+1]*0.75]
latent_class_interps += [latent_class_means[-1]]
latent_class_interps = torch.stack(latent_class_interps)

soft_one_hot = F.softmax(latent_class_interps,dim=2)
latent_class_interps_z = torch.einsum('b n c, c d -> b n d',soft_one_hot,model.embeddings.weight)
queries = model.posenc_dec_in(torch.zeros(latent_class_interps.shape[0], model.hparams.output_seq_len,latent_class_interps_z.shape[2])).to(latent_class_interps_z.device)
latent_class_interps_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_interps_z,tgt_mask=model.dec_mask))
latent_class_interps_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_interps_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_interps_output.shape[0],latent_class_interps_output.shape[1],22*3+3)

pose = latent_class_interps_output_smpl_repr[:,:,3:]
trans = latent_class_interps_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_interps",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())


# %%
bwd_w = torch.linspace(0,1,latent_class_means.shape[1]).unsqueeze(1)
fwd_w = torch.linspace(1,0,latent_class_means.shape[1]).unsqueeze(1)
latent_class_time_interps = []
for i in range(latent_class_means.shape[0]-1):
    latent_class_time_interps += [latent_class_means[i],
                            latent_class_means[i]*fwd_w+latent_class_means[i+1]*bwd_w]
latent_class_time_interps += [latent_class_means[-1]]
latent_class_time_interps = torch.stack(latent_class_time_interps)

soft_one_hot = F.softmax(latent_class_time_interps,dim=2)
latent_class_time_interps_z = torch.einsum('b n c, c d -> b n d',soft_one_hot,model.embeddings.weight)
queries = model.posenc_dec_in(torch.zeros(latent_class_time_interps.shape[0], model.hparams.output_seq_len,latent_class_time_interps_z.shape[2])).to(latent_class_time_interps_z.device)
latent_class_time_interps_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_time_interps_z,tgt_mask=model.dec_mask))
latent_class_time_interps_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_time_interps_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_time_interps_output.shape[0],latent_class_time_interps_output.shape[1],22*3+3)

pose = latent_class_time_interps_output_smpl_repr[:,:,3:]
trans = latent_class_time_interps_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_time_interps",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())

# %%

latent_class_means = torch.stack([torch.from_numpy(np.mean(z_all.reshape(z_all.shape[0],
                        model.hparams.output_seq_len,-1)[seq_indices[x]],axis=0)) for x in ["run","sit","stand","jump","turn"]])
latent_class_means_z = latent_class_means
queries = model.posenc_dec_in(torch.zeros(latent_class_means.shape[0], model.hparams.output_seq_len,latent_class_means_z.shape[2])).to(latent_class_means_z.device)
latent_class_means_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_means_z,tgt_mask=model.dec_mask))
latent_class_means_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_means_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_means_output.shape[0],latent_class_means_output.shape[1],22*3+3)

pose = latent_class_means_output_smpl_repr[:,:,3:]
trans = latent_class_means_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_means_post_softmax",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())

#%%
latent_class_interps = []
for i in range(latent_class_means.shape[0]-1):
    latent_class_interps += [latent_class_means[i],
                            latent_class_means[i]*0.75+latent_class_means[i+1]*0.25,
                            latent_class_means[i]*0.5+latent_class_means[i+1]*0.5,
                            latent_class_means[i]*0.25+latent_class_means[i+1]*0.75]
latent_class_interps += [latent_class_means[-1]]
latent_class_interps = torch.stack(latent_class_interps)

latent_class_interps_z = latent_class_interps
queries = model.posenc_dec_in(torch.zeros(latent_class_interps.shape[0], model.hparams.output_seq_len,latent_class_interps_z.shape[2])).to(latent_class_interps_z.device)
latent_class_interps_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_interps_z,tgt_mask=model.dec_mask))
latent_class_interps_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_interps_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_interps_output.shape[0],latent_class_interps_output.shape[1],22*3+3)

pose = latent_class_interps_output_smpl_repr[:,:,3:]
trans = latent_class_interps_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_interps_post_softmax",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())


# %%
bwd_w = torch.cat([torch.tensor([0]*5), torch.linspace(0,1,latent_class_means.shape[1]-10), torch.tensor([1]*5)]).unsqueeze(1)
fwd_w = torch.cat([torch.tensor([1]*5), torch.linspace(1,0,latent_class_means.shape[1]-10), torch.tensor([0]*5)]).unsqueeze(1)
latent_class_time_interps = []
for i in range(latent_class_means.shape[0]-1):
    latent_class_time_interps += [latent_class_means[i],
                            latent_class_means[i]*fwd_w+latent_class_means[i+1]*bwd_w]
latent_class_time_interps += [latent_class_means[-1]]
latent_class_time_interps = torch.stack(latent_class_time_interps)


latent_class_time_interps_z = latent_class_time_interps
queries = model.posenc_dec_in(torch.zeros(latent_class_time_interps.shape[0], model.hparams.output_seq_len,latent_class_time_interps_z.shape[2])).to(latent_class_time_interps_z.device)
latent_class_time_interps_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=latent_class_time_interps_z,tgt_mask=model.dec_mask))
latent_class_time_interps_output_smpl_repr = nmg_repr.nmg2smpl((latent_class_time_interps_output*dset.data_std + dset.data_mean).reshape(-1,22,9)).reshape(latent_class_time_interps_output.shape[0],latent_class_time_interps_output.shape[1],22*3+3)

pose = latent_class_time_interps_output_smpl_repr[:,:,3:]
trans = latent_class_time_interps_output_smpl_repr[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"_".join(selected_sequences)+"latent_time_interps_post_softmax",
                    root_orient=pose[:,:,:3].detach().cpu().numpy(),
                    pose_body=pose[:,:,3:].detach().cpu().numpy(),
                    trans=trans.detach().cpu().numpy())


# %% K-means
from sklearn.cluster import KMeans
z_init_centroids = []
for seq in selected_sequences:
    z_init_centroids.append(z_all[seq_indices[seq]].mean(axis=0))
z_init_centroids = np.array(z_init_centroids)
z_km = KMeans(n_clusters=len(selected_sequences),init=z_init_centroids).fit(z_all)

ip_init_centroids = []
for seq in selected_sequences:
    ip_init_centroids.append(inputs.reshape(inputs.shape[0],-1)[seq_indices[seq]].mean(axis=0))
ip_init_centroids = np.array(ip_init_centroids)
ip_km = KMeans(n_clusters=len(selected_sequences),init=ip_init_centroids).fit(inputs.reshape(inputs.shape[0],-1))

babel_labels = np.zeros(z_all.shape[0])
for id,seq in enumerate(selected_sequences):
    babel_labels[seq_indices[seq]] = id