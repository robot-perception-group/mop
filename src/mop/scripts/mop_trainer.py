import matplotlib
from ray.tune.suggest import search
matplotlib.use("Agg")

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
import itertools

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

import sys
idx_num = int(sys.argv[1])

hparams = yaml.safe_load(open(osp.join(config.nmg_repo_path,"src/nmg/scripts/hparams.yaml")))

search_params_keys = [x for x in hparams if type(hparams[x]) is list]
search_params_vals = list(itertools.product(*[hparams[x] for x in hparams if type(hparams[x]) is list]))

for i,k in enumerate(search_params_keys):
    hparams[k] = search_params_vals[idx_num][i]


version = "v{}_".format(idx_num) + ",".join([str(search_params_keys[i])+"="+str(search_params_vals[idx_num][i]) for i in range(len(search_params_keys))])


if osp.exists(osp.join(config.logdir, hparams["name"], version)):

    print("##########################")
    print("resuming from "+osp.join(config.logdir,hparams["name"], version))
    print("##########################")

    hparams = yaml.safe_load(open(osp.join(config.logdir,hparams["name"],version,"hparams.yaml")))

    pl.seed_everything(seed=hparams["PL_GLOBAL_SEED"])

    model = nmg.nmg(hparams)
    num_gpus=1


    logger = TensorBoardLogger(save_dir=config.logdir, name=hparams["name"],version=version)

    if osp.exists(osp.join(config.logdir, hparams["name"],version ,"checkpoints","last.ckpt")):
        trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                                        gpus=num_gpus,
                                        resume_from_checkpoint=osp.join(config.logdir, hparams["name"],version ,"checkpoints","last.ckpt"),
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="rec_pose_loss/val",mode="min",save_top_k=1,save_last=True)])
    else:
        trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                                        gpus=num_gpus,
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="rec_pose_loss/val",mode="min",save_top_k=1,save_last=True)])
                                        
    trainer.fit(model)

else:
    
    if hparams["PL_GLOBAL_SEED"] is None:
        pl.seed_everything()
        hparams["PL_GLOBAL_SEED"] = int(os.environ["PL_GLOBAL_SEED"])
    else:
        pl.seed_everything(int(hparams["PL_GLOBAL_SEED"]))

    model = nmg.nmg(hparams)
    num_gpus=1


    logger = TensorBoardLogger(save_dir=config.logdir, name=hparams["name"],version=version)
    
    os.makedirs(osp.join(config.logdir,hparams["name"],version))

    yaml.dump(hparams, open(osp.join(config.logdir,hparams["name"],version,"hparams.yaml"),"w"))
    
    trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                                        gpus=num_gpus,
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="rec_pose_loss/val",mode="min",save_top_k=1,save_last=True)])
    trainer.fit(model)