# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from torchvision import models
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM





class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class ViTForImageReconstruction(nn.Module):
    def __init__(self, base_vit_model, decoder_embed_dim=512, image_size=224, *args, **kwargs):
        super().__init__()
        self.vit = base_vit_model(*args, **kwargs)  # Initialize the base ViT model

        @property
        def patch_embed(self):
            return self.vit.patch_embed  # Assuming `self.vit` is your base Vision Transformer model

        # Add the expansion and decoder layers
        self.expand_dim = nn.Linear(1024, decoder_embed_dim * 16 * 16)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # Resize feature map to 14x14 before decoder starts

        # Decoder: series of ConvTranspose2d layers to upsample the feature representation to the original image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output 1 channel
            nn.Sigmoid()  # Or another suitable activation function
        )

    def forward(self, x):
        # Encoder
        x = self.vit.forward_features(x)

        # Expand dimensions to prepare for the decoder
        x = self.expand_dim(x)
        x = x.view(-1, 512, 16, 16)  # Reshape to match decoder's expected input shape

        # Resize feature map to 14x14 to prepare for upsampling to 224x224
        x = self.adaptive_pool(x)

        # Decoder
        x = self.decoder(x)  # Upsample to original image size
        return x


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, reconstructed, target):
        loss_mse = self.mse(reconstructed, target)
        loss_mae = self.mae(reconstructed, target)
        return loss_mse + loss_mae  # You can adjust the weighting of these losses

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()

        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.to(device)  # Move VGG to the correct device
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.ssim_module = SSIM(data_range=255, size_average=True, channel=1, nonnegative_ssim=True)
        self.ssim_module.to(device)

    def forward(self, input, target):
        # Convert grayscale images to RGB
        input_rgb = input.repeat(1, 3, 1, 1)  # Assuming input is [N, 1, H, W]
        target_rgb = target.repeat(1, 3, 1, 1)  # Assuming target is [N, 1, H, W]

        # Perceptual Loss
        perceptual_loss = nn.functional.mse_loss(self.vgg(input_rgb), self.vgg(target_rgb))

        # SSIM Loss
        ssim_loss = 1 - self.ssim_module(input, target)

        # Combine losses
        final_loss = perceptual_loss + ssim_loss
        return final_loss




# Define the base Vision Transformer model
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


