from functools import partial
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.models.layers import DropPath, to_2tuple
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer

from crossattn_module import KD_Cross_Block


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.BatchNorm2d(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class Attention_module(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=1,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 fp_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.channel_expand = 4
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = DWConv(dim, dim * self.channel_expand)
        self.k = DWConv(dim, dim * self.channel_expand)
        self.v = DWConv(dim, dim)
        self.attn_conv_0 = nn.Sequential(
            nn.Conv2d(self.channel_expand, self.channel_expand, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_expand))
        self.attn_conv_1 = nn.Conv2d(self.channel_expand, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x):
        B, C, H, W = x.shape

        # new qkv
        q = self.q(x).reshape(B,  self.channel_expand, C, -1).transpose(-1, -2)
        k = self.k(x).reshape(B,  self.channel_expand, C, -1)
        v = self.v(x).reshape(B, 1, C, -1).transpose(-1, -2)

        attn = (q @ k) * self.scale
        attn = self.attn_conv_0(attn)
        attn_scores = torch.softmax(attn, dim=-1)
        attn = attn * attn_scores
        attn = self.attn_conv_1(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2).reshape(B, C, H, W)

        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Conv_module(nn.Module):
    def __init__(self,
                 dim,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        reduction_ratio = 4
        hidden_features = dim // reduction_ratio
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class HybridTokenMixer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 fp_size=0,
                 reduction_ratio=8):
        super().__init__()

        self.local_unit = Conv_module(dim=dim // 2)
        self.global_unit = Attention_module(dim=dim // 2, num_heads=num_heads, fp_size=fp_size)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.BatchNorm2d(inner_dim),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), )

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x
        return x


class Mlp(nn.Module):  ### FFN
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim=64,
                 num_heads=1,
                 mlp_ratio=4,
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 fp_size=0,):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.token_mixer = HybridTokenMixer(dim,
                                            num_heads=num_heads,
                                            fp_size=fp_size)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop, )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward_impl(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        x = self._forward_impl(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=1024,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 drop_rate=0,
                 num_classes=4,
                 layers=[2, 2, 2, 2]):

        super().__init__()
        self.layers = layers
        embed_dims = [48, 96, 224, 448]
        self.num_heads = [1, 1, 1, 1]
        mlp_ratios = [4, 4, 4, 4]
        self.fp_size = [64, 16, 8, 4]

        # Stage-0
        self.patch_embed_A0 = PatchEmbed(img_size=img_size,
                                         patch_size=32,
                                         in_chans=in_chans,
                                         embed_dim=embed_dims[0])
        self.block_A0 = nn.Sequential(*[
            Block(embed_dims[0], num_heads=self.num_heads[0], mlp_ratio=mlp_ratios[0],
                  act_cfg=act_cfg, drop=drop_rate, drop_path=0, fp_size=self.fp_size[0])
            for i in range(self.layers[0])
        ])

        # Stage-1
        self.patch_embed_A1 = PatchEmbed(img_size=img_size // 32,
                                         patch_size=2,
                                         in_chans=embed_dims[0],
                                         embed_dim=embed_dims[1])
        self.block_A1 = nn.Sequential(*[
            Block(embed_dims[1], num_heads=self.num_heads[1], mlp_ratio=mlp_ratios[1],
                  act_cfg=act_cfg, drop=drop_rate, drop_path=0, fp_size=self.fp_size[1])
            for i in range(self.layers[1])
        ])

        # Stage-2
        self.patch_embed_A2 = PatchEmbed(img_size=img_size // 64,
                                         patch_size=2,
                                         in_chans=embed_dims[1],
                                         embed_dim=embed_dims[2])
        self.block_A2 = nn.Sequential(*[
            Block(embed_dims[2], num_heads=self.num_heads[2], mlp_ratio=mlp_ratios[2],
                  act_cfg=act_cfg, drop=drop_rate, drop_path=0, fp_size=self.fp_size[2])
            for i in range(self.layers[2])
        ])

        # Stage-3
        self.patch_embed_A3 = PatchEmbed(img_size=img_size // 128,
                                             patch_size=2,
                                             in_chans=embed_dims[2],
                                             embed_dim=embed_dims[3])
        self.block_A3 = nn.Sequential(*[
            Block(embed_dims[3], num_heads=self.num_heads[3], mlp_ratio=mlp_ratios[3],
                  act_cfg=act_cfg, drop=drop_rate, drop_path=0, fp_size=self.fp_size[3])
            for i in range(self.layers[3])
        ])

        # kd cross attention
        self.kd_cross_attn_0 = KD_Cross_Block(dim=embed_dims[0], num_heads=1, fp_size=self.fp_size[0])
        self.kd_cross_attn_1 = KD_Cross_Block(dim=embed_dims[1], num_heads=1, fp_size=self.fp_size[1])
        self.kd_cross_attn_2 = KD_Cross_Block(dim=embed_dims[2], num_heads=1, fp_size=self.fp_size[2])


        # Classifier
        self.avg_final = nn.Sequential(
            build_norm_layer(norm_cfg, embed_dims[3])[1],
            nn.AdaptiveAvgPool2d(1)
        )
        path_dim = 128
        self.fc_new1 = nn.Sequential(nn.Linear(embed_dims[3], path_dim),
                                     nn.BatchNorm1d(path_dim),
                                     nn.ReLU(inplace=True)
                                     )
        self.fc_new2 = nn.Linear(path_dim, num_classes)

        self.apply(self._init_model_weights)

    # init for image classification
    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, x_a):
        x_a0 = self.patch_embed_A0(x_a)
        x_a0 = self.block_A0(x_a0)
        cross_a0, self.attn_a0 = self.kd_cross_attn_0(x_a0)
        f0 = cross_a0.flatten(1)

        x_a1 = self.patch_embed_A1(cross_a0)
        x_a1 = self.block_A1(x_a1)
        cross_a1, self.attn_a1 = self.kd_cross_attn_1(x_a1)
        f1 = cross_a1.flatten(1)

        x_a2 = self.patch_embed_A2(cross_a1)
        x_a2 = self.block_A2(x_a2)
        cross_a2, self.attn_a2 = self.kd_cross_attn_2(x_a2)
        f2 = cross_a2.flatten(1)

        x_a3 = self.patch_embed_A3(cross_a2)
        x_a3 = self.block_A3(x_a3)
        f3 = self.avg_final(x_a3).flatten(1)

        feat = self.fc_new1(f3)
        out = self.fc_new2(feat)

        return [f0, f1, f2, f3, feat], out
