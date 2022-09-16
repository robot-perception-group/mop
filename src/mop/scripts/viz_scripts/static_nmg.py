import bpy
from mathutils import *
from math import radians
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
cmap = plt.get_cmap("Blues")

D = bpy.data
C = bpy.context



empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
empty.rotation_euler[0] = math.radians(90)
empty.location[2] = 1.16

# data = np.load("/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/2021-11-06_conv_nmg/train_nmg_52d4c_00003_3_codebook_size=1000,gumbel_temp_anneal_rate=0.005,min_gumbel_softmax_temp=0.5_2021-11-06_20-42-45/checkpoints/epoch=1039-step=329679 copyrun_sit_stand_jump_jog_kick_turnlatent_time_interps.npz")

# data = np.load("/home/nsaini/Dropbox/cvpr22_mat/AMASS_sample400.npz")
data = np.load("/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/2021-11-06_conv_nmg/train_nmg_52d4c_00000_0_codebook_size=500,gumbel_temp_anneal_rate=0.001,min_gumbel_softmax_temp=0.5_2021-11-06_20-39-57/checkpoints/epoch=1849-step=586449/latent_interps2seq.npz")


bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {}

num_samples = range(data["trans"].shape[0])
frames = range(0,data["trans"].shape[1],5)
locs_x, locs_z = np.meshgrid(range(0,8,2),range(0,8,2))
locs_x = locs_x.reshape(-1)
locs_z = locs_z.reshape(-1)

objs = bpy.data.objects
try:
    objs.remove(objs["Cube"], do_unlink=True)
    objs.remove(objs["Light"], do_unlink=True)
except:
    pass

# Create light datablock
light_data = bpy.data.lights.new(name="light", type='AREA')
light_data.energy = 10000
light_data.size = 25

# Create new object, pass the light data 
light_object = bpy.data.objects.new(name="light", object_data=light_data)

# Link object to collection in context
bpy.context.collection.objects.link(light_object)

# Change light position
light_object.location = (7, 6, 6)

objs["Camera"].location = (8,-19,8)
objs["Camera"].rotation_euler = (radians(70),0,radians(6))

for idx in num_samples:
    empty2 = D.objects.new("empty"+str(idx),None)
    C.scene.collection.objects.link(empty2)
    empty2.parent = empty
    for i,n in enumerate(frames):
        smpl_out = bm.forward(trans=torch.from_numpy(data["trans"][idx]).float(),
                                root_orient=torch.from_numpy(data["root_orient"][idx]).float(),
                                pose_body=torch.from_numpy(data["pose_body"][idx]).float())

        smpl_mesh = D.meshes.new("smpl_mesh_"+str(idx)+"_"+str(n))
        smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
        smpl_mesh.from_pydata(smpl_out.v[n].detach().numpy(),[],list(smpl_out.f.detach().numpy()))
        mat = bpy.data.materials.new("mat")
        mat.diffuse_color = (cmap(n/data["trans"].shape[1])[0],
                            cmap(n/data["trans"].shape[1])[1],
                            cmap(n/data["trans"].shape[1])[2],1)
                            # 0.8 + 0.2/(len(frames) - i))
        smpl_obj.active_material = mat
        C.scene.collection.objects.link(smpl_obj)
        smpl_obj.parent = empty2
        empty2.location[0] = locs_x[idx]
        empty2.location[2] = locs_z[idx]



# # Settings
# name = 'Gridtastic'
# rows = 5
# columns = 10
# size = 1

# # Utility functions
# def vert(column, row):
#     """ Create a single vert """

#     return (column * size, row * size, 0)


# def face(column, row):
#     """ Create a single face """

#     return (column* rows + row,
#            (column + 1) * rows + row,
#            (column + 1) * rows + 1 + row,
#            column * rows + 1 + row)

# # Looping to create the grid
# verts = [vert(x, y) for x in range(columns) for y in range(rows)]
# faces = [face(x, y) for x in range(columns - 1) for y in range(rows - 1)]

# # Create Mesh Datablock
# mesh = bpy.data.meshes.new(name)
# mesh.from_pydata(verts, [], faces)

# # Create Object and link to scene
# obj = bpy.data.objects.new(name, mesh)
# bpy.context.scene.collection.objects.link(obj)
