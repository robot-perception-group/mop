#%%
import numpy as np
from nmg.dsets import amass, hact12
from nmg.models import nmg
import os
import os.path as osp
from pytorch_lightning import callbacks, profiler
import pytorch_lightning
from tqdm import tqdm
import pytorch_lightning as pl
import yaml
from nmg.scripts import ray_trainer
from tqdm import tqdm
import torch
from nmg.utils import transforms as p3dt
import matplotlib.pyplot as plt
import torch.nn as nn
from nmg.utils import config
%pylab inline

home_dir = "/is/ps3/nsaini"

#%% Load pretrained VQNMG and check embedding viz
from nmg.utils import nmg_repr
ckpt_path = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/neural-motion-graph/nmg_logs/Trial_0310_gumbel_softmax_vq_little_amass/train_nmg_25b18_00002_2_gumbel_temp_anneal_rate=0.001,kl_annealing=False,kl_annealing_cycle_epochs=100,latent_len=10,lr=1.3987e-05_2021-10-03_17-05-46/checkpoints/epoch=3911-step=50855.ckpt"
model = nmg.nmg.load_from_checkpoint(ckpt_path)
# hparams = yaml.load(open(osp.join(home_dir,"projects/neural-motion-graph/src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
# model = nmg.nmg(hparams)
model.eval()

output_seq_len = 60

train_dset, val_dset = amass.get_amass_train_test(model.hparams)



# inputs = [val_dset[np.random.randint(0,len(val_dset))] for _ in range(5)]
# file_name_string = "val_reconstruction"

# # inputs = [train_dset[np.random.randint(0,len(train_dset))] for _ in range(5)]
# # file_name_string = "train_reconstruction"

# dsample = torch.stack([x["data"] for x in inputs])
# encoder_output, decoder_output, z = model(dsample)
# decoder_output = torch.cat([dsample,decoder_output])


# M = 10*model.hparams.latent_len
# np_one_hot = np.zeros((M,model.hparams.codebook_size))
# np_one_hot[range(M),np.random.choice(model.hparams.codebook_size,M)] = 1
# np_one_hot = np.reshape(np_one_hot,[-1,model.hparams.latent_len,model.hparams.codebook_size])
# one_hot = torch.from_numpy(np_one_hot).to(model.device).float()
# z = torch.einsum('b n c, c d -> b n d',one_hot,model.embeddings.weight)
# queries = model.posenc_dec_in(torch.zeros(z.shape[0], output_seq_len,z.shape[2])).to(z.device)
# decoder_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=z,tgt_mask=model.dec_mask))
# file_name_string = "latent_samples"

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


#%% Load pretrained nmg model and test 
ckpt_path = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/neural-motion-graph/nmg_logs/Trial_0309_with_conts_mse_loss/train_nmg_13606_00002_2_conts_loss_weight=1,conts_speed_loss_weight=0.01,gt_conts_for_speed_loss=True,kl_annealing_cycle_epochs=20_2021-09-03_09-41-14/checkpoints/epoch=999-step=251999.ckpt"
model = nmg.nmg.load_from_checkpoint(ckpt_path)
# hparams = yaml.load(open(osp.join(home_dir,"projects/neural-motion-graph/src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
# model = nmg.nmg(hparams)
model.eval()

train_dset, val_dset = amass.get_amass_train_test(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"),seq_begin=None,input_seq_len=60,output_seq_len=90)

viz = []
viz_names = []
def hook_fn(m,input,output):
    viz.append(output)
    viz_names.append(m)

for i in range(model.hparams.num_layers):
    model.nmg_enc.layers[i].self_attn.register_forward_hook(hook_fn)
for i in range(model.hparams.num_layers):
    model.nmg_dec.layers[i].self_attn.register_forward_hook(hook_fn)
for i in range(model.hparams.num_layers):
    model.nmg_dec.layers[i].multihead_attn.register_forward_hook(hook_fn)


output_seq_len = 90

inputs = [val_dset[np.random.randint(0,len(val_dset))],val_dset[np.random.randint(0,len(val_dset))]]

dsample = torch.stack([x["data"] for x in inputs])
if model.hparams.use_contacts:
    contacts = torch.stack([torch.from_numpy(x["contacts"]) for x in inputs])
    dsample = torch.cat([dsample[:,:,:3],torch.cat([dsample[:,:,3:].reshape(dsample.shape[0],-1,22,6),contacts[:,:dsample.shape[1]].unsqueeze(-1)],dim=3).reshape(dsample.shape[0],-1,22*7)],dim=2)
encout,_ = model.forward(dsample)

# z = torch.randn(4,1024).float().to(model.device)
z = torch.cat([encout[:,0],
                encout[:,0]+0.5*torch.rand(1,encout.shape[2]).float().to(encout.device),
                encout[:,0]+torch.rand(1,encout.shape[2]).float().to(encout.device),
                encout[:,0]+1.5*torch.rand(1,encout.shape[2]).float().to(encout.device)],dim=0)

batch_size = z.shape[0]
queries = model.posenc_dec_in(torch.zeros(batch_size, output_seq_len,z.shape[1])).to(z.device)
decoder_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=z.unsqueeze(1)))
if model.hparams.use_contacts:
    decoder_output_conts = decoder_output[:,:,3:].reshape(batch_size,output_seq_len,22,7)[:,:,:,6]
    decoder_output_pose = decoder_output[:,:,3:].reshape(batch_size,output_seq_len,22,7)[:,:,:,:6].reshape(batch_size,-1,22*6)
else:
    decoder_output_conts = None
    decoder_output_pose = decoder_output[:,:,3:]
decoder_output_pose = (decoder_output_pose*train_dset.data_std[3:])+train_dset.data_mean[3:]
decoder_output_trans = (decoder_output[:,:,:3]*train_dset.data_std[:3])+train_dset.data_mean[:3]
pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(decoder_output_pose.view(batch_size,output_seq_len,22,6))).view(batch_size,output_seq_len,-1)
trans = decoder_output_trans

np.savez(".".join(ckpt_path.split(".")[:-1])+"val_cont_rand_morelen_latent_sample",
            root_orient=pose[:,:,:3].detach().cpu().numpy(),
            pose_body=pose[:,:,3:].detach().cpu().numpy(),
            contacts=torch.sigmoid(decoder_output_conts).detach().cpu().numpy(),
            trans=trans.detach().cpu().numpy())


#%% Visualization code check
hparams = yaml.load(open(osp.join(home_dir,"projects/neural-motion-graph/src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
model = nmg.nmg(hparams)
amass_dset = amass.amass(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"))
data = np.load(amass_dset.npz_files[0])
smpl_out = model.bm(trans=torch.from_numpy(data["trans"]).float(),
                    root_orient=torch.from_numpy(data["poses"][:,:3]).float(),
                    pose_body=torch.from_numpy(data["poses"][:,3:66]).float())
from nmg.utils import viz
fig = viz.viz_smpl_seq(smpl_out.v,smpl_out.f)

#%% AMASS train test 120 and 60 fps test
train_dset, val_dset = amass.get_amass_train_test(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"))
low_lengths = []
for idx in tqdm(range(len(val_dset))):
    sample = val_dset[idx]
    if sample["data"].shape[0] < 100:
        low_lengths.append(sample["data"].shape[0])
        print(val_dset.npz_files[idx])

#%% AMASS 100 len and 120, 60 fps data generation
amass_dset = amass.amass(osp.join(config.dataset_path,"amass_april_2019/npz_files_100len_fps_120_60.pkl"))
high_lengths = []
for idx in tqdm(range(len(amass_dset))):
    sample = amass_dset[idx]
    if sample["data"].shape[0] >= 100:
        high_lengths.append(amass_dset.npz_files[idx])

#%% pytorch lightning single instance training
hparams = yaml.load(open(osp.join(config.nmg_repo_path,"src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
hparams["PL_GLOBAL_SEED"] = int(os.environ["PL_GLOBAL_SEED"])

model = nmg.nmg(hparams)
num_gpus=1

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger(config.logdir, name=hparams["name"])

trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                            gpus=num_gpus,
                            logger=logger,
                            track_grad_norm=2,profiler="simple")
trainer.fit(model)



#%%
# read files from the dataset and check the frame rate
npz_files = []
count=0
for root,_,files in os.walk(osp.join(config.dataset_dir,"amass_april_2019/")):
    for fls in files:
        if ".npz" in fls:
            sample = np.load(osp.join(root,fls))
            try:
                if sample["mocap_framerate"] == 60 and sample["trans"].shape[0] >=100: 
                    npz_files.append(osp.join(root,fls))
                elif sample["mocap_framerate"] == 120 and sample["trans"].shape[0] >=200: 
                    npz_files.append(osp.join(root,fls))
            except:
                print(osp.join(root,fls))
                count += 1
print(count)

pkl.dump(npz_files,open(osp.join(config.dataset_dir,"amass_april_2019/100len_fps_120_60.pkl"),"wb"))

#%%
# get mean and std of the dataset
dataset_path = "/home/nsaini/Datasets/amass_april_2019/100len_fps_120_60.pkl"
dset = amass.amass({"datapath":dataset_path,"seq_begin":None,"input_seq_len":90,"output_seq_len":90,"mean_std_path":None})
means = []
stds = []
for m in tqdm(range(5)):
    samples = torch.stack([dset[x]["data"] for x in tqdm(range(len(dset)))])




# %% Amass PCA
train_dset, validation_dset = amass.get_amass_train_test(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"),seq_begin=0,seq_len=60)
torch_dset = torch.stack([train_dset[i]["data"] for i in tqdm(range(len(train_dset)))])
torch_val_dset = torch.stack([validation_dset[i]["data"] for i in tqdm(range(len(validation_dset)))])

dset = torch_dset.view(5046,-1).T.data.cpu().numpy()
dset_mean = np.mean(dset,1)
dset_std = dset-dset_mean[:,np.newaxis]
cov_mat = np.cov(dset_std)
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
plt.step(range(0,len(cum_var_exp)), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)


latent_dim = 2

w = np.hstack([eigen_pairs[i][1][:,np.newaxis] for i in range(latent_dim)])
proj_mat = np.matmul(w,w.T)

# train data reconstruction
rec_dset = np.matmul(proj_mat,dset_std) + dset_mean[:,np.newaxis]
# save reconstructed sample
torch_sample = torch.from_numpy(rec_dset[:,0]).float().reshape(60,-1)
trans = torch_sample[:,:3]
pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(torch_sample[:,3:].view(60,-1,6))).view(60,-1)
np.savez("amass_pca_recon_sample_latent1024",
            root_orient=pose[:,:3].detach().numpy(),
            pose_body=pose[:,3:].detach().numpy(),
            trans=trans.detach().numpy())
# save original sample
orig_torch_sample = torch.from_numpy(dset[:,0]).float().reshape(60,-1)
orig_trans = orig_torch_sample[:,:3]
orig_pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(orig_torch_sample[:,3:].view(60,-1,6))).view(60,-1)
np.savez("amass_pca_orig_sample_latent1024",
            root_orient=orig_pose[:,:3].detach().numpy(),
            pose_body=orig_pose[:,3:].detach().numpy(),
            trans=orig_trans.detach().numpy())

# test data reconstruction
val_dset = torch_val_dset.view(1262,-1).T.data.cpu().numpy()
val_dset_std = val_dset-dset_mean[:,np.newaxis]
val_rec_dset = np.matmul(proj_mat,val_dset_std) + dset_mean[:,np.newaxis]
# save reconstructed sample
val_torch_sample = torch.from_numpy(val_rec_dset[:,0]).float().reshape(60,-1)
trans = val_torch_sample[:,:3]
pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(val_torch_sample[:,3:].view(60,-1,6))).view(60,-1)
np.savez("amass_pca_recon_test_sample_latent1024",
            root_orient=pose[:,:3].detach().numpy(),
            pose_body=pose[:,3:].detach().numpy(),
            trans=trans.detach().numpy())
# save original sample
orig_val_torch_sample = torch.from_numpy(val_dset[:,0]).float().reshape(60,-1)
orig_val_trans = orig_val_torch_sample[:,:3]
orig_val_pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(orig_val_torch_sample[:,3:].view(60,-1,6))).view(60,-1)
np.savez("amass_pca_orig_test_sample_latent1024",
            root_orient=orig_val_pose[:,:3].detach().numpy(),
            pose_body=orig_val_pose[:,3:].detach().numpy(),
            trans=orig_val_trans.detach().numpy())

# %%
smplout = bm.forward(root_orient=orient,pose_body=pose,betas=betas)
loss = torch.sum(smplout.Jtr)
import timeit
start_time = timeit.default_timer()
loss.backward()
elapsed_time = timeit.default_timer() - start_time




# %%
from nmg.utils.preprocess import preprocess_data
babel_root_dir = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/datasets/babel_v1.0_release"
amass_root_dir = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/datasets/body_data"
result_dir = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/datasets/amass_babel_temp"
preprocess_data(babel_root_dir, amass_root_dir, result_dir, -1)




# %% TSNE of the learned latent space

ckpt_path = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/neural-motion-graph/nmg_logs/Trial_1808_frame_mean_std_dataAug/train_nmg_e7a47_00003_3_kl_annealing_cycle_epochs=10,lr=4.1719e-05_2021-08-17_20-44-24/checkpoints/epoch=999-step=251999.ckpt"
model = nmg.nmg.load_from_checkpoint(ckpt_path)
# hparams = yaml.load(open(osp.join(home_dir,"projects/neural-motion-graph/src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
# model = nmg.nmg(hparams)
model.eval()

train_dset, val_dset = amass.get_amass_train_test(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"),seq_begin=0,seq_len=60)

seq_len = 60

train_means = []
frame_labels = []

from collections import Counter
for idx in tqdm(range(1000)):
    dsample = train_dset[idx]
    count = Counter(dsample["frame_anns"])
    frame_labels.append(count.most_common(1)[0][0])
    encout,_ = model.forward(dsample["data"].unsqueeze(0))
    train_means.append(encout[0,0,:])

t_means = torch.stack(train_means)
unique_classes = np.unique(frame_labels)
from matplotlib import cm
clrs = [cm.viridis(x) for x in np.linspace(0,1,len(unique_classes))]
colmap = dict(zip(unique_classes,clrs))

colors = [colmap[x] for x in frame_labels]

from sklearn.manifold import TSNE
tsne_means = TSNE(2).fit_transform(t_means.data.cpu().numpy())



# %% Find treadmill sequence
dset = amass.amass(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"),seq_begin=0,input_seq_len=None,output_seq_len=None)
for fl in tqdm(dset.npz_files):
    da = np.load(fl)
    if "run on treadmill" in np.char.lower(da["frame_action"]) or \
        "walk on treadmill" in np.char.lower(da["frame_action"]) or \
            "jog on treadmill" in np.char.lower(da["frame_action"]):
        print(fl)


# %% generate long sequence in autoregressive way by outputting longer sequence than input sequence
ckpt_path = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/neural-motion-graph/nmg_logs/Trial_2508_more_output_len/train_nmg_cd400_00004_4_kl_annealing_cycle_epochs=20,lr=2.405e-05_2021-08-25_10-32-53/checkpoints/epoch=999-step=251999.ckpt"
model = nmg.nmg.load_from_checkpoint(ckpt_path)
# hparams = yaml.load(open(osp.join(home_dir,"projects/neural-motion-graph/src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
# model = nmg.nmg(hparams)
model.eval()

train_dset, val_dset = amass.get_amass_train_test(osp.join(home_dir,"datasets/amass_babel/npz_files_100len_fps_120_60.pkl"),seq_begin=None,input_seq_len=60,output_seq_len=90)


output_seq_len = 90

dsample = torch.stack([val_dset[40]["data"],val_dset[90]["data"]])
encout,_ = model.forward(dsample)

# z = torch.randn(4,1024).float().to(model.device)
z = torch.cat([encout[:,0],
                encout[:,0]+0.5*torch.rand(1,encout.shape[2]).float().to(encout.device),
                encout[:,0]+torch.rand(1,encout.shape[2]).float().to(encout.device),
                encout[:,0]+1.5*torch.rand(1,encout.shape[2]).float().to(encout.device)],dim=0)

batch_size = z.shape[0]
queries = model.posenc_dec_in(torch.zeros(batch_size, output_seq_len,z.shape[1])).to(z.device)
decoder_output = model.dec_lin(model.nmg_dec(tgt=queries,memory=z.unsqueeze(1)))

dec_out = [decoder_output]
dec_out_stacked = decoder_output.detach()
for id in range(10):
    next_in = dec_out[-1][:,30:]
    inp_orient,inp_trans = amass.transform_seq_init(next_in[:,:,:3],next_in[:,:,3:].reshape(-1,60,22,6)[:,:,0,:],repr="6d") 
    inp = torch.cat([inp_trans,p3dt.matrix_to_rotation_6d(inp_orient),next_in[:,:,9:]],dim=2)
    z = model.forward(inp)[0][:,0]
    next_out = model.dec_lin(model.nmg_dec(tgt=queries,memory=z.unsqueeze(1)))
    dec_out.append(next_out)
    out_orient,out_trans = amass.transform_seq_init(dec_out[-1][:,:,:3],
                                                    dec_out[-1][:,:,3:].reshape(-1,output_seq_len,22,6)[:,:,0,:],
                                                    repr="6d",
                                                    target_rot=p3dt.rotation_6d_to_matrix(dec_out_stacked[:,-60,3:9]),
                                                    target_trans=dec_out_stacked[:,-60,:3])
    
    out = torch.cat([out_trans,p3dt.matrix_to_rotation_6d(out_orient),dec_out[-1][:,:,9:]],dim=2)
    dec_out_stacked = torch.cat([dec_out_stacked[:,:-60],out],dim=1)

dec_out_stacked = (dec_out_stacked*train_dset.data_std)+train_dset.data_mean
pose = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(dec_out_stacked[:,:,3:3+22*6].view(batch_size,dec_out_stacked.shape[1],22,6))).view(batch_size,dec_out_stacked.shape[1],-1)
trans = dec_out_stacked[:,:,:3]

np.savez(".".join(ckpt_path.split(".")[:-1])+"val_cont_rand_morelen_autoreg_latent_sample",
            root_orient=pose[:,:,:3].detach().numpy(),
            pose_body=pose[:,:,3:].detach().numpy(),
            trans=trans.detach().numpy())

dec_out_debug = torch.stack([dec_out[i][1] for i in range(len(dec_out))])
dec_out_debug = (dec_out_debug*train_dset.data_std)+train_dset.data_mean
pose_debug = p3dt.matrix_to_axis_angle(p3dt.rotation_6d_to_matrix(dec_out_debug[:,:,3:3+22*6].view(dec_out_debug.shape[0],dec_out_debug.shape[1],22,6))).view(dec_out_debug.shape[0],dec_out_debug.shape[1],-1)
trans_debug = dec_out_debug[:,:,:3]
np.savez(".".join(ckpt_path.split(".")[:-1])+"autoreg_debug",
            root_orient=pose_debug[:,:,:3].detach().numpy(),
            pose_body=pose_debug[:,:,3:].detach().numpy(),
            trans=trans_debug.detach().numpy())
# %%
#%% Load pretrained nmg model and test 
ckpt_path = "/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/projects/neural-motion-graph/nmg_logs/Trial_3008_with_contacts/train_nmg_1d44b_00004_4_conts_speed_loss_weight=0.001,kl_annealing_cycle_epochs=20,lr=3.4814e-05_2021-08-30_20-37-39/checkpoints/epoch=999-step=251999.ckpt"
model = nmg.nmg.load_from_checkpoint(ckpt_path)
model.eval()
_, val_dset = amass.get_amass_train_test(model.hparams.datapath,seq_begin=model.hparams.seq_begin,input_seq_len=model.hparams.input_seq_len,output_seq_len=model.hparams.output_seq_len)

from torch.utils.data import DataLoader
dl = DataLoader(val_dset,batch_size=model.hparams.batch_size,
                            num_workers=model.hparams.num_wrkrs,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)
trainer = pl.Trainer(gpus=1)

trainer.test(model,dl,ckpt_path)


#%% little amass

# Running sequences
trials = ["09_"+'{:02d}'.format(x) for x in [1,2,3,4,5,6,7,8,9,10,11]]
trials += ["35_"+'{:02d}'.format(x) for x in [17,18,19,20,21,22,23,24,25,26]]
trials += ["141_"+'{:02d}'.format(x) for x in [1,2,3,34]]
trials += ["143_"+'{:02d}'.format(x) for x in [1,42]]
trials += ["16_"+'{:02d}'.format(x) for x in [35,36,45,46,55,56,]]

# walk sequences
trials += ["02_"+'{:02d}'.format(x) for x in [1,2]]
trials += ["07_"+'{:02d}'.format(x) for x in [1,2,3,4,5,6,7,8,9,10,11]]
trials += ["08_"+'{:02d}'.format(x) for x in [1,2,3,4,6,8,9,10]]
trials += ["16_"+'{:02d}'.format(x) for x in [15,16,21,22,31,32,47,58]]
trials += ["20_"+'{:02d}'.format(x) for x in [4,5]]
trials += ["21_"+'{:02d}'.format(x) for x in [4,5]]
trials += ["35_"+'{:02d}'.format(x) for x in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,28,29,30,31,32,33,34]]
trials += ["39_"+'{:02d}'.format(x) for x in [1,2,3,4,5,6,7,8,9,10,12,13,14]]
trials += ["69_"+'{:02d}'.format(x) for x in [1,2,3,4,5]]
trials += ["136_"+'{:02d}'.format(x) for x in [20,21,22,23,24]]

import pickle as pkl
amass_list = pkl.load(open("/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/datasets/amass_babel/npz_files_100len_fps_120_60.pkl","rb"))
amass_cmu = [x for x in amass_list if "CMU" in x]
prefix = "/".join(amass_cmu[0].split("/")[:-2])
amass_cmu = set(["_".join(x.split("/")[-1].split("_")[:-1]) for x in amass_cmu])
trials_set = set(trials)

trials = list(amass_cmu.intersection(trials_set))
import os
little_amass_npz_files_100len_fps_120_60 = [os.path.join(prefix,x.split("_")[0],x+"_poses.npz") for x in trials]
pkl.dump(little_amass_npz_files_100len_fps_120_60,open("/trainman-mount/trainman-k8s-storage-8e3fd382-2f06-466b-8bde-ca27be865ff5/datasets/amass_babel/little_amass_npz_files_100len_fps_120_60.pkl","wb"))