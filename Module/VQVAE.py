import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from adamp import AdamP
from omegaconf import OmegaConf
import yaml
import os, sys
from einops import rearrange, reduce, asnumpy, parse_shape
from einops.layers.torch import Rearrange, Reduce

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.residual_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, 3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 3, stride=1, padding=1, bias=False)
        )

        self.residual_stack = nn.ModuleList([ self.residual_block for i in range(self.num_residual_layers)])            

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = x + self.residual_stack[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            #Rearrange('b c h w -> b (c h w)'),
            nn.Conv2d(in_channels, num_hiddens // 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens // 2, num_hiddens, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, 3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):   
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens, 3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens // 2, 3, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.embedding = nn.Embedding(self.K, self.D) # k d
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)
        self.beta = beta

    def forward(self, x):
        x = rearrange(x, 'b d h w -> b h w d') # b c h w == b d h w
        x_flat = rearrange(x, 'b h w d -> (b h w) d')

        # L2 distances
        distances = torch.sum(x_flat ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight ** 2, dim=1) - \
                    2 * torch.einsum('ik,kj->ij', [x_flat, self.embedding.weight.t()]) # bhw k

        # indices encoding
        encoding_indices = rearrange(torch.argmin(distances, dim=1),'bhw -> bhw ()')
        # one hot encoding
        encoding_one_hot = torch.zeros(encoding_indices.size(0) ,self.K, device=x.device) # bhw k
        encoding_one_hot.scatter_(1, encoding_indices, 1) # bhw k

        # quantize latents 
        x_quantized = torch.einsum('ik,kj->ij', [encoding_one_hot, self.embedding.weight]) # bhw, d
        x_quantized = rearrange(x_quantized, '(b h w) d -> b h w d', **parse_shape(x, 'b h w d'))

        # Loss
        commitment_loss = F.mse_loss(x_quantized.detach(), x)
        codebook_loss = F.mse_loss(x_quantized, x.detach())
        vq_loss = codebook_loss + commitment_loss * self.beta

        x_quantized = x + (x_quantized - x).detach()
        avg_probs = reduce(encoding_one_hot, 'bhw k -> k', 'mean')
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return vq_loss, codebook_loss, commitment_loss, rearrange(x_quantized, 'b h w d -> b d h w'), perplexity, encoding_one_hot

# class VectorQuantizerEMA(nn.Module):

class VQVAE(pl.LightningModule):
    
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, beta, lr, cifar_train_variance):
        super().__init__()
        self.encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.pre_conv = nn.Conv2d(num_hiddens, embedding_dim, 1, stride=1)

        self.lr = lr
        self.beta = beta
        self.cifar_train_variance = cifar_train_variance

        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_conv(z)
        vq_loss, codebook_loss, commitment_loss, x_quantized, perplexity, encoding_one_hot = self.vq(z)
        x_hat = self.decoder(x_quantized)
        return vq_loss, codebook_loss, commitment_loss, x_hat, perplexity
    
    def configure_optimizers(self):
        optimizer = AdamP(self.parameters(),  lr=self.lr)
        return optimizer
    
    def loss_function(self, vq_loss, codebook_loss, commitment_loss, x_hat, x):
        reconstructin_loss = F.mse_loss(x_hat, x) / self.cifar_train_variance
        loss = reconstructin_loss + vq_loss
        #print(f"loss:{loss} reconstructin_loss: {reconstructin_loss}, codebook_loss:{codebook_loss}, commitment_loss:{commitment_loss}")
        self.log('reconstructin_loss', reconstructin_loss)
        self.log('codebook_loss', codebook_loss)
        self.log('commitment_loss', commitment_loss)
        return loss
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        vq_loss, codebook_loss, commitment_loss, x_hat, perplexity = self.forward(x)
        #print(f"[ {self.__class__.__name__} {sys._getframe().f_code.co_name} ] x_hat: {x_hat.shape}")
        loss = self.loss_function(vq_loss, codebook_loss, commitment_loss, x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        vq_loss, codebook_loss, commitment_loss, x_hat, perplexity = self.forward(x)
        loss = self.loss_function(vq_loss, codebook_loss, commitment_loss, x_hat, x)
        self.log('val_loss', loss)
        return x_hat

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        vq_loss, codebook_loss, commitment_loss, x_hat, perplexity = self.forward(x)
        loss = self.loss_function(vq_loss, codebook_loss, commitment_loss, x_hat, x)
        self.log('test_loss', loss)
        return x_hat

    def validation_epoch_end(self, validation_step_outputs):
        for pred in validation_step_outputs:
            self.logger.experiment.log(
                {"valid/img": [wandb.Image(img) for img in (pred.detach())],
                 "global_step": self.global_step})
            break
        
    def test_epoch_end(self, outputs):
        print(type(outputs))
        print(len(outputs))
        # print(type(outputs)[0])
        # print(outputs[0].shape)
        for pred in outputs:
            grid_img = make_grid(pred.detach().cpu(), nrow=8)
            plt.figure()
            plt.imshow(grid_img.permute(1, 2, 0))
            break