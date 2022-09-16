import bpy
from mathutils import *
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math
from nmg.dsets import amass_humor
from nmg.utils.nmg_repr import nmg2smpl

D = bpy.data
C = bpy.context

amh = amass_humor.amass_humor()

data = []
for i in np.random.randint(0,len(amh),5):
    data.append(nmg2smpl(((amh[i]["data"]*amh.data_std) + amh.data_mean).reshape(amh.input_seq_len,22,9)).unsqueeze(0))

data = torch.cat(data)

empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
# empty.rotation_euler[0] = math.radians(90)
# empty.location[2] = 1.16

bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {}

motion_range = range(data.shape[0])
# motion_range = range(8,17)

for idx in motion_range:
    smpl_out[str(idx)] = bm.forward(trans=data[idx,:,:3],
                            root_orient=data[idx,:,3:6],
                            pose_body=data[idx,:,6:])
    # smpl_out[str(idx)] = bm.forward(trans=torch.zeros(30,3).float(),
    #                         root_orient=torch.zeros(30,3).float(),
    #                         pose_body=torch.from_numpy(data["pose_body"][idx]).float())

    smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
    smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
    smpl_mesh.from_pydata(smpl_out[str(idx)].v[0].detach().numpy(),[],list(smpl_out[str(idx)].f.detach().numpy()))
    C.scene.collection.objects.link(smpl_obj)
    smpl_obj.parent = empty
    smpl_obj.location[0] = idx

def anim_handler(scene):
    frame=scene.frame_current
    
    for idx in motion_range:
        ob = D.objects.get("smpl_mesh"+str(idx))
        ob.data.clear_geometry()
        ob.data.from_pydata(smpl_out[str(idx)].v[frame].detach().numpy(),[],list(smpl_out[str(idx)].f.detach().numpy()))

bpy.app.handlers.frame_change_pre.append(anim_handler)