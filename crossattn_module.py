import torch.nn as nn
import torch
from einops import rearrange
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer, ConvModule


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


class Cross_Attention_v3(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 fp_size=0,
                 qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.channel_expand = 4

        self.q_a = DWConv(dim, dim)
        self.k_a = DWConv(dim, dim * self.channel_expand)
        self.v_a = DWConv(dim, dim)
        self.attn_conv_a0 = nn.Sequential(
            nn.Conv2d(self.channel_expand, self.channel_expand, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_expand))
        self.attn_conv_a1 = nn.Conv2d(self.channel_expand, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_a = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.q_b = DWConv(dim, dim)
        self.k_b = DWConv(dim, dim * self.channel_expand)
        self.v_b = DWConv(dim, dim)
        self.attn_conv_b0 = nn.Sequential(
            nn.Conv2d(self.channel_expand, self.channel_expand, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_expand))
        self.attn_conv_b1 = nn.Conv2d(self.channel_expand, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_b = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x_a, x_b):
        B, C, H, W = x_a.shape

        q_a = self.q_a(x_a).reshape(B, 1, C, -1).transpose(-1, -2)
        k_a = self.k_a(x_a).reshape(B, self.channel_expand, C, -1)
        v_a = self.v_a(x_a).reshape(B, 1, C, -1).transpose(-1, -2)

        q_b = self.q_b(x_b).reshape(B, 1, C, -1).transpose(-1, -2)
        k_b = self.k_b(x_b).reshape(B, self.channel_expand, C, -1)
        v_b = self.v_b(x_b).reshape(B, 1, C, -1).transpose(-1, -2)

        attn_a = (q_b @ k_a) * self.scale  # [B, self.channel_expand, C, C]
        attn_a = self.attn_conv_a0(attn_a)
        attn_scores_a = torch.softmax(attn_a, dim=-1)
        attn_a = attn_a * attn_scores_a
        attn_a = self.attn_conv_a1(attn_a)
        out_a = (attn_a @ v_a).transpose(-1, -2).reshape(B, C, H, W)
        out_a = self.proj_a(out_a)

        attn_b = (q_a @ k_b) * self.scale
        attn_b = self.attn_conv_b0(attn_b)
        attn_scores_b = torch.softmax(attn_b, dim=-1)
        attn_b = attn_b * attn_scores_b
        attn_b = self.attn_conv_b1(attn_b)
        out_b = (attn_b @ v_b).transpose(-1, -2).reshape(B, C, H, W)
        out_b = self.proj_b(out_b)

        return out_a, out_b, attn_a, attn_b


class Mlp(nn.Module):  ### FFN
    def __init__(self,
                 in_features,
                 mlp_ratio=4,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Cross_Block(nn.Module):
    def __init__(self,
                 dim,
                 fp_size,
                 num_heads):
        super().__init__()
        # self.norm1_a = nn.LayerNorm([dim, fp_size, fp_size])
        # self.norm1_b = nn.LayerNorm([dim, fp_size, fp_size])
        self.norm1_a = nn.BatchNorm2d(dim)
        self.norm1_b = nn.BatchNorm2d(dim)
        self.cross_attn = Cross_Attention_v3(dim, num_heads=num_heads, fp_size=fp_size)
        # self.norm2_a = nn.LayerNorm([dim, fp_size, fp_size])
        # self.norm2_b = nn.LayerNorm([dim, fp_size, fp_size])
        self.norm2_a = nn.BatchNorm2d(dim)
        self.norm2_b = nn.BatchNorm2d(dim)
        self.mlp_a = Mlp(dim)
        self.mlp_b = Mlp(dim)

    def forward(self, x_a, x_b):
        out_a, out_b, attn_a, attn_b = self.cross_attn(self.norm1_a(x_a), self.norm1_b(x_b))
        out_a = out_a + x_a
        out_b = out_b + x_b
        out_a = out_a + self.mlp_a(self.norm2_a(out_a))
        out_b = out_b + self.mlp_b(self.norm2_b(out_b))

        return out_a, out_b, attn_a, attn_b


class KD_Cross_Attention_v3(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 qk_scale=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.channel_expand = 4

        self.q_a = DWConv(dim, dim)
        self.k_a = DWConv(dim, dim * self.channel_expand)
        self.v_a = DWConv(dim, dim)
        self.attn_conv_a0 = nn.Sequential(
            nn.Conv2d(self.channel_expand, self.channel_expand, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_expand))
        self.attn_conv_a1 = nn.Conv2d(self.channel_expand, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_a = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x_a):
        B, C, H, W = x_a.shape

        q_a = self.q_a(x_a).reshape(B, 1, C, -1).transpose(-1, -2)
        k_a = self.k_a(x_a).reshape(B, self.channel_expand, C, -1)
        v_a = self.v_a(x_a).reshape(B, 1, C, -1).transpose(-1, -2)

        attn_a = (q_a @ k_a) * self.scale  # [B, self.channel_expand, C, C]
        attn_a = self.attn_conv_a0(attn_a)
        attn_scores_a = torch.softmax(attn_a, dim=-1)
        attn_a = attn_a * attn_scores_a
        attn_a = self.attn_conv_a1(attn_a)
        out_a = (attn_a @ v_a).transpose(-1, -2).reshape(B, C, H, W)
        out_a = self.proj_a(out_a)

        return out_a, attn_a


class KD_Cross_Block(nn.Module):
    def __init__(self,
                 dim,
                 fp_size,
                 num_heads, ):
        super().__init__()

        # self.norm1_a = nn.LayerNorm([dim, fp_size, fp_size])
        self.norm1_a = nn.BatchNorm2d(dim)
        self.kd_cross_attn = KD_Cross_Attention_v3(dim, num_heads=num_heads)
        self.norm2_a = nn.BatchNorm2d(dim)
        self.mlp_a = Mlp(dim)

    def forward(self, x_a):
        out_a, attn_a = self.kd_cross_attn(self.norm1_a(x_a))
        out_a = out_a + x_a
        out_a = out_a + self.mlp_a(self.norm2_a(x_a))

        return out_a, attn_a
