import pytorch_lightning as pl
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from ..dsets import amass, hact12, amass_humor
from torch.utils.data import DataLoader
from torch.nn import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import math
import numpy as np
from ..utils import transforms as p3dt
from ..utils.fid import calculate_frechet_distance
from torch.autograd import Variable
from human_body_prior.body_model.body_model import BodyModel
from ..models import stgcn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ..utils import config


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.1)

def create_mask(dim,attn_frame_len,symmetric):
    mask = torch.ones(dim,dim).float()
    for id in range(dim):
        zero_len = (id-attn_frame_len) if (id-attn_frame_len) > 0 else 0
        mask[id][:zero_len] = 0
        one_len = (id+attn_frame_len+1) if (id+attn_frame_len+1) < dim else dim
        mask[id][one_len:] = 0
    
    if symmetric:
        return mask
    else:
        return torch.tril(mask)


class Residual(nn.Module):
    def __init__(self, latent_dim):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim,
                      out_channels=latent_dim,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv1d(in_channels=latent_dim,
                      out_channels=latent_dim,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class mop(pl.LightningModule):
    
    def __init__(self, hparams):
        '''
        '''
        
        super().__init__()
        
        self.save_hyperparameters(hparams)
        self.n_features = 9
    
        input_dim = 22*self.n_features
        latent_dim = self.hparams.latent_dim
        num_heads = self.hparams.num_heads
        dim_feedfwd = self.hparams.dim_feedfwd
        drpout = 0.1
        num_layers = self.hparams.num_layers
        codebook_size = self.hparams.codebook_size
        self.enc_lin3 = nn.Linear(latent_dim, codebook_size)

        if self.hparams.ae_type.lower() == "vae":
            ip_seq_len = self.hparams.input_seq_len+2

        self.nmg_enc = nn.Sequential(nn.Conv1d(input_dim,latent_dim,3,padding=1,stride=1),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU())
        self.nmg_enc_lin_mu = nn.Linear(self.hparams.latent_len*self.hparams.latent_dim,self.hparams.latent_dim)
        self.nmg_enc_lin_std = nn.Linear(self.hparams.latent_len*self.hparams.latent_dim,self.hparams.latent_dim)
        
        self.nmg_dec_lin = nn.Linear(self.hparams.latent_dim,self.hparams.latent_len*self.hparams.latent_dim)
        self.nmg_dec = nn.Sequential(Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            Residual(latent_dim),
                            nn.GELU(),
                            nn.ConvTranspose1d(latent_dim,input_dim,3,padding=1,stride=1))
        
        # Embeddings
        self.embeddings = nn.Embedding(self.hparams.codebook_size,latent_dim)

        # weights initialization
        self.apply(init_weights)

        # body model
        self.bm = BodyModel(osp.join(config.dataset_dir,"smpl_models/smplh/neutral/model.npz"))
        self.bm.eval()

        # kl weight initilaization for kl annealing and gumbel softmax temperature
        if self.hparams.kl_annealing:
            self.kl_weight = 0
        else:
            self.kl_weight = 1
        self.gumbel_softmax_temp = self.hparams.init_gumbel_softmax_temp

    def encode(self,enc_in):
        '''
        '''
        batch_size = enc_in.shape[0]
        encoder_output = self.nmg_enc(enc_in.permute(0,2,1)).view(enc_in.shape[0],-1)
        encoder_output = torch.stack([self.nmg_enc_lin_mu(encoder_output),self.nmg_enc_lin_std(encoder_output)],dim=1)

        return encoder_output

    def decode(self,z):
        batch_size = z.shape[0]
        decoder_output = self.nmg_dec(self.nmg_dec_lin(z).reshape(batch_size,-1,self.hparams.latent_dim).permute(0,2,1)).permute(0,2,1)
        return decoder_output

    def latent_sample(self,encoder_output):
        mu = encoder_output[:,0]
        logvar = encoder_output[:,1]
        std = torch.exp(logvar/2)
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu).unsqueeze(1)

        return z
        

    def forward(self,enc_in):
        '''
        '''
        
        encoder_output = self.encode(enc_in)
        z = self.latent_sample(encoder_output)
        decoder_output = self.decode(z)

        return encoder_output, decoder_output, z
        


    def fwd_pass_and_loss(self,batch,batch_idx):
        '''
        '''
        input_data = batch["data"]
        output_data = batch["output_data"]
        batch_size = input_data.shape[0]
        seq_len = input_data.shape[1]

        if self.hparams.use_contacts:
            net_in = torch.cat([input_data[:,:,:3],torch.cat([input_data[:,:,3:].reshape(batch_size,-1,22,self.n_features),batch["contacts"][:,:seq_len].unsqueeze(-1)],dim=3).reshape(batch_size,-1,22*(self.n_features+1))],dim=2)
        else:
            net_in = input_data

        # forward pass
        encoder_output, decoder_output, z = self.forward(net_in)

        
        # reconstruction loss
        rec_pose_loss = F.mse_loss(output_data,decoder_output)
        
        # KL div loss
        mu = encoder_output[:,0]
        logvar = encoder_output[:,1]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # loss combined
        loss = self.hparams.rec_pose_weight * rec_pose_loss +\
                self.kl_weight * kl_loss


        losses = {"loss": loss.detach(),
                    "rec_pose_loss": rec_pose_loss.detach(),
                    "kl_loss": kl_loss.detach()}

        output = {"encoder_output": encoder_output.detach(),
                    "decoder_output": decoder_output.detach(),
                    "z": z.detach()}

        return output, losses, loss



    def training_step(self,batch,batch_idx):
        output, losses, loss = self.fwd_pass_and_loss(batch,batch_idx)

        # if batch_idx % self.hparams.summary_steps == 0:
        #     for loss_name, val in losses.items():
        #         self.log(loss_name + '/train', val)

        return {"loss":loss, "output":output, "losses": losses}

    def validation_step(self,batch,batch_idx):
        with torch.no_grad():
            output, losses, loss = self.fwd_pass_and_loss(batch,batch_idx)

            # if batch_idx % self.hparams.summary_steps == 0:
            #     for loss_name, val in losses.items():
            #         self.log(loss_name + '/val', val)

            non_weighted_loss = 0
            for loss_name,val in losses.items():
                non_weighted_loss += val

        return {"val_loss": loss,"non_weighted_loss":non_weighted_loss, "rec_pose_loss":losses["rec_pose_loss"], "output":output, "losses":losses}

    def training_epoch_end(self, outputs):
        if self.hparams.kl_annealing:
            if (self.current_epoch+1) % (2*self.hparams.kl_annealing_cycle_epochs) == 0:
                self.kl_weight = 0
            elif (self.current_epoch+1) % (2*self.hparams.kl_annealing_cycle_epochs) <= self.hparams.kl_annealing_cycle_epochs:
                self.kl_weight += 1/self.hparams.kl_annealing_cycle_epochs
        elif self.hparams.ae_type.lower() == "autoenc":
            self.kl_weight = 0
        else:
            self.kl_weight = 1

        for loss_type in outputs[0]["losses"].keys():
            self.log(loss_type + "/train",torch.stack([x["losses"][loss_type] for x in outputs]).mean())

        self.gumbel_softmax_temp = np.max([self.hparams.init_gumbel_softmax_temp*np.exp(-self.hparams.gumbel_temp_anneal_rate*self.current_epoch),self.hparams.min_gumbel_softmax_temp])
        
        self.log("anneal_params/kl_weight",self.kl_weight)
        if self.hparams.ae_type.lower() == "vqvae":
            self.log("anneal_params/gumbel_temp",self.gumbel_softmax_temp)
        
        if self.hparams.tsne_log_freq is not None:
            if self.current_epoch % self.hparams.tsne_log_freq == 0:
                z = torch.cat([x["output"]["z"] for x in outputs],dim=0)
                z_wth_embeddgings = torch.cat([z.reshape(-1,self.hparams.latent_dim),self.embeddings.weight.detach()],dim=0)
                tsne_means = TSNE(2).fit_transform(z_wth_embeddgings.data.cpu().numpy())
                fig,ax = plt.subplots()
                ax.scatter(tsne_means[:z_wth_embeddgings.shape[0],0],tsne_means[:z.shape[0],1])
                ax.scatter(tsne_means[z_wth_embeddgings.shape[0]:,0],tsne_means[z.shape[0]:,1],c="k")

                self.logger.experiment.add_figure("latent_space/train",fig,self.current_epoch)

    def validation_epoch_end(self, outputs):
        non_weighted_avg_loss = torch.stack(
            [x["non_weighted_loss"] for x in outputs]).mean()
        self.log("ptl/non_weighted_val_loss", non_weighted_avg_loss)
        rec_pose_avg_loss = torch.stack(
            [x["rec_pose_loss"] for x in outputs]).mean()
        self.log("ptl/rec_pose_val_loss", rec_pose_avg_loss)
        
        for loss_type in outputs[0]["losses"].keys():
            self.log(loss_type + "/val",torch.stack([x["losses"][loss_type] for x in outputs]).mean())
        
        if self.hparams.tsne_log_freq is not None:
            if self.current_epoch % self.hparams.tsne_log_freq == 0:
                z = torch.cat([x["output"]["z"] for x in outputs],dim=0)
                z_wth_embeddgings = torch.cat([z.reshape(-1,self.hparams.latent_dim),self.embeddings.weight.detach()],dim=0)
                tsne_means = TSNE(2).fit_transform(z_wth_embeddgings.data.cpu().numpy())
                fig,ax = plt.subplots()
                ax.scatter(tsne_means[:z.shape[0],0],tsne_means[:z.shape[0],1])
                ax.scatter(tsne_means[z.shape[0]:,0],tsne_means[z.shape[0]:,1],c="k")

                self.logger.experiment.add_figure("latent_space/val",fig,self.current_epoch)



    def train_dataloader(self):

        if self.hparams.data.lower() == "hact12":
            train_dset, _ = hact12.get_hact12_train_test(self.hparams,cats=[1,2])
        elif self.hparams.data.lower() == "amass_humor":
            train_dset, _ = amass_humor.get_amass_humor_train_test(self.hparams)
        else:
            train_dset, _ = amass.get_amass_train_test(self.hparams)
            if self.hparams.overfit_mode:
                train_dset.npz_files = train_dset.npz_files[:self.hparams.batch_size]

        return DataLoader(train_dset,batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_wrkrs,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)



    def val_dataloader(self):
        if self.hparams.data.lower() == "hact12":
            _, val_dset = hact12.get_hact12_train_test(self.hparams,cats=[1,2])
        elif self.hparams.data.lower() == "amass_humor":
            _, val_dset = amass_humor.get_amass_humor_train_test(self.hparams)
        else:
            _, val_dset = amass.get_amass_train_test(self.hparams)
            if self.hparams.overfit_mode:
                val_dset.npz_files = val_dset.npz_files[:self.hparams.batch_size]

        return DataLoader(val_dset,batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_wrkrs,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)



    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                            gamma=self.hparams.lr_decay_rate)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}