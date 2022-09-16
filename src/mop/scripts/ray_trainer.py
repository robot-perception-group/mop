import matplotlib
from pytorch_lightning.utilities import seed
matplotlib.use("Agg")

from nmg.models import nmg
import pytorch_lightning as pl
from ray import tune
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import os.path as osp
import os
import yaml
from nmg.utils import config
from ray.tune import Stopper
import time
import sys
import numpy as np
from ray.tune.schedulers import ASHAScheduler


        

class dict2obj:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)

def train_nmg(hparams, num_gpus=1, resume=False):

    if resume and osp.isfile(osp.join(tune.get_trial_dir(),"checkpoints","last.ckpt")):
        hparams = yaml.load(open(osp.join(tune.get_trial_dir(),"hparams.yaml")))
        pl.seed_everything(hparams["PL_GLOBAL_SEED"])

    model = nmg.nmg(hparams)

    num_gpus = int(np.ceil(num_gpus))

    # create logger
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(),name="",version=".")

    metrics = {"rec_pose_loss": "ptl/rec_pose_val_loss"}

    if resume and osp.isfile(osp.join(tune.get_trial_dir(),"checkpoints","last.ckpt")):
        print("##########################")
        print("resuming trial "+tune.get_trial_dir())
        print("##########################")
        trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                            gpus=num_gpus,
                            logger=logger,
                            resume_from_checkpoint=osp.join(tune.get_trial_dir(),"checkpoints","last.ckpt"),
                            callbacks=[TuneReportCallback(metrics, on="validation_end"),
                                        ModelCheckpoint(monitor="rec_pose_loss/val",mode="min",save_top_k=1,save_last=True),
                                        LearningRateMonitor(logging_interval="epoch")],
                            progress_bar_refresh_rate=300,
                            check_val_every_n_epoch=hparams["check_val_every_n_epoch"])
    else:
        trainer = pl.Trainer(max_epochs=hparams["max_epochs"],
                                gpus=num_gpus,
                                logger=logger,
                                callbacks=[TuneReportCallback(metrics, on="validation_end"),
                                            ModelCheckpoint(monitor="rec_pose_loss/val",mode="min",save_top_k=1,save_last=True),
                                            LearningRateMonitor(logging_interval="epoch")],
                                progress_bar_refresh_rate=300,
                                check_val_every_n_epoch=hparams["check_val_every_n_epoch"])

    trainer.fit(model)



if __name__ == "__main__":

    hparams = yaml.load(open(osp.join(config.nmg_repo_path,"src/nmg/scripts/hparams.yaml")),Loader=yaml.FullLoader)
    

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(torch.cuda.device_count())])

    gpus_per_trial = 1
    cpus_per_trial = hparams["num_wrkrs"]
    num_samples = 1


    # hparams["activation"] = tune.choice(["relu", "gelu"])
    # hparams["lr"] = tune.loguniform(1e-5,1e-4)
    # hparams["latent_dim"] = tune.grid_search([64,1024])
    # hparams["straight_through"] = tune.grid_search([False, True])
    hparams["kl_annealing_cycle_epochs"] = tune.grid_search([10,30])
    hparams["rec_pose_weight"] = tune.grid_search([1,2,5])
    # hparams["rec_joint_positions_weight"] = tune.uniform(2,10)
    # hparams["rec_trans_weight"] = tune.uniform(2,10)
    # hparams["conts_speed_loss_weight"] = tune.grid_search([0.001,0.01,0.1,1])
    # hparams["conts_loss_weight"] = tune.grid_search([1,10])
    # hparams["gt_conts_for_speed_loss"] = tune.grid_search([True,False])

    # hparams["init_gumbel_softmax_temp"] = tune.grid_search([1,10])
    # hparams["min_gumbel_softmax_temp"] = tune.grid_search([0.5,0.1])
    # hparams["gumbel_temp_anneal_rate"] = tune.grid_search([0.001,0.005])
    # hparams["codebook_size"] = tune.grid_search([500,1000])

    # scheduler = ASHAScheduler(
    #     max_t=hparams["max_epochs"],
    #     grace_period=100,
    #     reduction_factor=2)

    if osp.exists(osp.join(config.logdir,hparams["name"])):
        print("##########################")
        print("resuming from "+osp.join(config.logdir,hparams["name"]))
        print("##########################")

        analysis = tune.run(tune.with_parameters(train_nmg,num_gpus=gpus_per_trial,resume=True),
                            resources_per_trial={"cpu":cpus_per_trial,"gpu":gpus_per_trial},
                            metric="rec_pose_loss",
                            mode="min",
                            config=hparams,
                            num_samples=num_samples,
                            local_dir=config.logdir,
                            name=hparams["name"],
                            resume="LOCAL",
                            server_port=4279)
    else:
        if hparams["PL_GLOBAL_SEED"] is None:
            pl.seed_everything()
            hparams["PL_GLOBAL_SEED"] = int(os.environ["PL_GLOBAL_SEED"])
        else:
            pl.seed_everything(int(hparams["PL_GLOBAL_SEED"]))
        analysis = tune.run(tune.with_parameters(train_nmg,num_gpus=gpus_per_trial,resume=False),
                            resources_per_trial={"cpu":cpus_per_trial,"gpu":gpus_per_trial},
                            metric="rec_pose_loss",
                            mode="min",
                            config=hparams,
                            num_samples=num_samples,
                            local_dir=config.logdir,
                            name=hparams["name"],
                            server_port=4277)
