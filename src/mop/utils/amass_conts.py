import torch
import os
import os.path as osp
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel

home_dir = os.environ["SENSEI_USERSPACE_SELF"]

# create SMPLs
smplm = BodyModel(osp.join(home_dir,"datasets/smpl_models/amass_body_models/smplh/male/model.npz"))
smplf = BodyModel(osp.join(home_dir,"datasets/smpl_models/amass_body_models/smplh/female/model.npz"))
smpln = BodyModel(osp.join(home_dir,"datasets/smpl_models/amass_body_models/smplh/neutral/model.npz"))

# SMPL contact joint candidates
j_indices = [1,2,5,8,4,7,18,20,19,21]

# load form AMASS
import numpy as np

amass = []
for root,dirs,files in os.walk(osp.join(home_dir,"datasets/amass_babel")):
    for fls in files:
        if (".npz" in fls) and (root.split("/")[-1] != "amass_babel"):
            amass.append(osp.join(root,fls))


for seq_no in tqdm(range(len(amass))):
# for seq_no in tqdm(range(1)):
    try:
        
        # load amass sequence
        seq = np.load(amass[seq_no])
        
        # get thetas (body joints only)
        thetas = torch.from_numpy(seq['poses'][:,:72]).float()
        thetas[:,66:] = 0
        
        # get betas
        betas = torch.from_numpy(seq['betas'][:10]).float().unsqueeze(0)
        
        # get root translation
        trans = torch.from_numpy(seq['trans']).float()
        
        # SMPL forward with gender specific model
        if str(seq['gender']).lower() == 'male':
            out = smplm.forward(trans=trans,
                             pose_body=thetas[:,3:66],
                             root_orient=thetas[:,:3],
                             betas=betas)
        elif str(seq['gender']).lower() == 'female':
            out = smplf.forward(trans=trans,
                             pose_body=thetas[:,3:66],
                             root_orient=thetas[:,:3],
                             betas=betas)
        else:
            out = smpln.forward(trans=trans,
                             pose_body=thetas[:,3:66],
                             root_orient=thetas[:,:3],
                             betas=betas)
        
        verts = out.v
        j_pos = out.Jtr[:,:24,:]
        
        # calculate velocities (speed) of the joints
        j_vel = j_pos[1:]-j_pos[:-1]
        j_speed = torch.sqrt(torch.sum(j_vel**2,2))

        ######################################
        # Probability of joint j in contact with the environment at any time t,
        # conditioned on its speed and z value is given as:
        # P(j_t|z_t,v_t) = P(j_t|z_t) * P(j_t|v_t)
        # the individual probabilities are modeled as:
        # P(j_t|z_t) = 1 - (z_t - min_j(z_t)) / (max_j(z_t) - min_j(z_t))
        # P(j_t|v_t) = - log10(v_t + 0.1)
        ######################################

        # P(J_t/z_t)
        p_z = 1- (j_pos[:,:,2].data.numpy()-np.min(j_pos[:,:,2].data.numpy(),axis=1)[:,np.newaxis]
              )/(np.max(j_pos[:,:,2].data.numpy(),axis=1)-np.min(j_pos[:,:,2].data.numpy(),axis=1))[:,np.newaxis]
        
        # allocate contacts array
        conts = np.zeros([j_pos.shape[0],24,4])
        
        # Last three columns are joint positions
        conts[:,:,:3] = j_pos.data.numpy()
        
        # contact probabilities in a temporary array
        temp = -np.log10(j_speed.data.numpy()+0.1)*p_z[1:]    # temp = (1/(j_speed.data.numpy()+1)**4)*p_z[1:]
        
        # thresholding
        max_idx = temp>0.9
        
        # assign final values
        conts[1:,:,3][max_idx] = temp[max_idx]
        
        # save results
        np.savez(amass[seq_no],**seq,contacts=conts[:,:,3])
    
    except:
        pass
        print(amass[seq_no])
        
import ipdb; ipdb.set_trace()
# For visualization
########################################################
smpl_anim.smpl_anim_w_conts(trans=trans.data.numpy(),shape=betas.data.numpy(),pose=thetas.data.numpy(),conts=conts[2:,j_indices])
# smpl_anim.smpl_anim(trans=trans.data.numpy(),shape=betas.data.numpy(),pose=thetas.data.numpy())
