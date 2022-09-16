# Adapted from Luyang's preprocess.py in vposer_analysis
import sys, os
import os.path as osp
import json
from collections import defaultdict, Counter

import numpy as np
import torch

from tqdm import tqdm

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


def get_action_categories(ann, file):
    # Get sequence labels and frame labels if they exist
    frame_cats = []
    if 'extra' not in file:
        if ann['frame_ann'] is not None:
            frame_cats = [(seg['act_cat'], seg['start_t'], seg['end_t']) for seg in ann['frame_ann']['labels']]
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['frame_anns'] is not None:            
            frame_cats = [(seg['act_cat'], seg['start_t'], seg['end_t']) for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']]
            
    return frame_cats

def preprocess_data(babel_root_dir, amass_root_dir, result_dir, cnt_thre):
    # load BABEL Dataset
    l_babel_dense_files = ['train', 'val']
    l_babel_extra_files = ['extra_train', 'extra_val']
    babel = {}
    for file in l_babel_dense_files:
        babel[file] = json.load(open(osp.join(babel_root_dir, file+'.json')))

    # for file in l_babel_extra_files:
    #     babel[file] = json.load(open(osp.join(babel_root_dir, file+'.json')))

    # preprocessing AMASS+BABEL
    cnt = 0
    # iterate over splits
    for spl in babel:
        # iterate over motion sequences
        for sid in tqdm(babel[spl]):
            # get current babel annotation
            cur_ann = babel[spl][sid]
            
            # load amass data
            amass_path = osp.join(amass_root_dir, '/'.join(cur_ann['feat_p'].split('/')[1:]))
            # print(amass_path)
            amass_data = np.load(amass_path)
            fps = float(amass_data["mocap_framerate"])
            
            # init per frame action annotations
            frame_action = ['null' for i in range(amass_data['poses'].shape[0])]
            # get per frame action annotations
            # We use action categories (~200) instead of proc_labels(~6000) to reduce action number
            frame_cats = get_action_categories(cur_ann, spl)
            # import ipdb;ipdb.set_trace()
            for cur_cat_info in frame_cats:
                start_frame = max(0, int(fps * cur_cat_info[1]))
                end_frame = min(int(fps * cur_cat_info[2]), amass_data['poses'].shape[0])
                for idx in range(start_frame, end_frame):
                    try:
                        if cur_cat_info[0] is not None:
                            frame_action[idx] = ','.join(cur_cat_info[0])
                        else:
                            frame_action[idx] = "null"
                    except:
                        import ipdb;ipdb.set_trace()
        
            # run vposer encoder inference given pose_body (root_orient not included),
            # output is torch.distributions.normal.Normal
            # amass_body_pose = torch.Tensor(amass_data['poses'][:, 3:66]).float().to(comp_device)
            # amass_body_poZ = vp.encode(amass_body_pose)
            # amass_body_poZ_mean = amass_body_poZ.mean.cpu().numpy()
            # amass_body_poZ_stddev = amass_body_poZ.stddev.cpu().numpy()
            # print(amass_body_pose.shape[0], amass_body_poZ_mean.shape, amass_body_poZ_stddev.shape)
            
            # save results
            result_path = osp.join(result_dir, '/'.join(cur_ann['feat_p'].split('/')[1:]))
            os.makedirs(osp.dirname(result_path), exist_ok=True)
            """
            Structure of npz file in AMASS dataset is as follows.
            - trans (num_frames, 3):  translation (x, y, z) of root joint
            - gender str: Gender of actor
            - mocap_framerate int: Framerate in Hz
            - betas (16): Shape parameters of body. See https://smpl.is.tue.mpg.de/
            - dmpls (num_frames, 8): DMPL parameters
            - poses (num_frames, 156): Pose data. Each pose is represented as 156-sized
                array. The mapping of indices encoding data is as follows:
                0-2 Root orientation
                3-65 Body joint orientations
                66-155 Finger articulations
            - frame_action (num_frames, ): per frame action annotation. 
            Each frame may have multiple action categories
            according BABEL dataset. If a frame has multiple action annotations, it is splited by ',' ('ACT_A,ACT_B,...').
                If a frame does not have action annotation, it is 'null'.
            - poZ_mean (num_frames, 32): mean value of latent distribution computed from vposer encoder.
            - poZ_stddev (num_frames, 32): stddev value of latent distribution computed from vposer encoder.
            """
            np.savez(
                    result_path,
                    frame_action=frame_action,
                    **amass_data
                )
            cnt += 1
            if cnt_thre != -1 and cnt > cnt_thre:
                break

if __name__ == '__main__':
    # directory setting
    babel_root_dir = 'babel_v1.0_release'
    amass_root_dir = 'amass/body_data'
    body_model_dir = 'smpl_models/amass_body_models/smplh'
    vposer_ckpt_dir = 'V02_05'
    result_dir = 'amass_babel'
    # computation device
    comp_device = torch.device("cuda:0")
    preprocess_data(babel_root_dir, amass_root_dir, body_model_dir, vposer_ckpt_dir, result_dir, comp_device)
