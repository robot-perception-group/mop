import bpy
from mathutils import *
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math

D = bpy.data
C = bpy.context


def delete_hierarchy(parent_obj_name):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects[parent_obj_name]
    obj.animation_data_clear()
    names = set()
    
    def get_child_names(obj):
        for child in obj.children:
            names.add(child.name)
            if child.children:
                get_child_names(child)

    get_child_names(obj)

    print(names)
    objects = bpy.data.objects
    [objects[n].select_set(True) for n in names]
    # Remove the animation from the all the child objects
    for child_name in names:
        bpy.data.objects[child_name].animation_data_clear()

    result = bpy.ops.object.delete()
    if result == {'FINISHED'}:
        print ("Successfully deleted object")
    else:
        print ("Could not delete object")


# delete_hierarchy("empty")



empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
# empty.rotation_euler[0] = math.radians(90)
# empty.location[2] = 1.16

data = np.load("/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs/conv_nmg_vae_211221-193320/nmg_logs/conv_nmg_vae_2021_12_15/train_nmg_fea95_00004_4_kl_annealing_cycle_epochs=10,rec_pose_weight=5_2021-12-16_03-19-39/checkpoints/epoch=869-step=275789/rand_samples_sigma2.npz")

bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {}

motion_range = range(data["trans"].shape[0])
# motion_range = range(8,17)

for idx in motion_range:
    smpl_out[str(idx)] = bm.forward(trans=torch.from_numpy(data["trans"][idx]).float(),
                            root_orient=torch.from_numpy(data["root_orient"][idx]).float(),
                            pose_body=torch.from_numpy(data["pose_body"][idx]).float())
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