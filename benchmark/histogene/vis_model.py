# Vendored / adapted from https://github.com/maxpmx/HisToGene (MIT)
# HisToGene Lightning module only (abundance head via n_genes=1).

from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import ViT


class HisToGene(pl.LightningModule):
    def __init__(
        self,
        patch_size=112,
        n_layers=4,
        n_genes=1000,
        dim=1024,
        learning_rate=1e-5,
        dropout=0.1,
        n_pos=64,
        use_checkpoint=False,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(
            dim=dim,
            depth=n_layers,
            heads=16,
            mlp_dim=2 * dim,
            dropout=dropout,
            emb_dropout=dropout,
            use_checkpoint=use_checkpoint,
        )
        self.gene_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))
        self.n_genes = n_genes

    def forward(self, patches, centers):
        # patches: [B, N, 3*H*W], centers: [B, N, 2] long
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        return self.gene_head(h)

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        return parser
