from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
from customized_linear import CustomizedLinear
from einops import rearrange
import random
import numpy as np
import pandas as pd
from kan import KAN
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_, to_2tuple, use_fused_attn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


        #
        self.attention_encoding = nn.TransformerEncoderLayer(embed_dim, nhead=1)
        #self.positional_encoding = nn.Embedding()
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)
        self.position_encoding = SinusoidalPositionalEncoding(embed_dim)
        self.sc = nn.Parameter(torch.tensor(0.001))
        self.sc2 = nn.Parameter(torch.tensor(0.2))
    def forward(self, x):
        num_cells = x.shape[0]
        #h = x
        x_fe = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        h = x_fe
        x_pos = rearrange(x, 'h (w c) -> h c w', c=self.num_patches)
        if x_pos.size(2) != 48:
            pad_size = 48 - x_pos.size(2)
            x_pos = torch.nn.functional.pad(x_pos, (0, pad_size))
        x_pos = self.position_encoding(x_pos)
        x_fe = x_fe + x_pos * self.sc
        #positions = torch.arange(self.num_patches, device=x.device).unsqueeze(0).expand(num_cells, -1)
        #pos_encoding = self.position_embedding(positions)
       # x = x + 0.1 * pos_encoding
        #h = x
        x = self.norm(x_fe)
        y = self.attention_encoding(h)
        y = self.norm(y)
        x = self.sc2 * y + x
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Attention1(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 local_window_size=2):  # Modify local_window_size to 2
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.local_window_size = local_window_size  # Initialize local_window_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Global Attention
        attn_global = (q @ k.transpose(-2, -1)) * self.scale
        attn_global = attn_global.softmax(dim=-1)
        attn_global = self.attn_drop(attn_global)
        x_global = (attn_global @ v).transpose(1, 2).reshape(B, N, C)

        # Local Attention
        q_local = q.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, C // self.num_heads)
        k_local = k.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, C // self.num_heads)
        v_local = v.permute(0, 2, 1, 3).reshape(B * self.num_heads, N, C // self.num_heads)

        x_local = torch.zeros_like(q_local)

        for i in range(0, N, self.local_window_size):
            q_win = q_local[:, i:i + self.local_window_size, :]
            k_win = k_local[:, i:i + self.local_window_size, :]
            v_win = v_local[:, i:i + self.local_window_size, :]
            attn_local = (q_win @ k_win.transpose(-2, -1)) * self.scale
            attn_local = attn_local.softmax(dim=-1)
            attn_local = self.attn_drop(attn_local)
            x_local[:, i:i + self.local_window_size, :] = (attn_local @ v_win)

        x_local = x_local.reshape(B, self.num_heads, N, C // self.num_heads).permute(0, 2, 1, 3).reshape(B, N, C)

        # Combine Global and Local Attention
        x = x_global + x_local

        x = self.proj(x)
        x = self.proj_drop(x)

        # Use global attention weights for output to keep it consistent with the original output format
        weights = attn_global.detach()

        return x, weights


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

        self.linear = nn.Linear(301, 144)
        self.linear1 = nn.Linear(144, 301)
        self.linear2 = nn.Linear(144, 301)
        self.linear3 = nn.Linear(144, 301)
    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        x = x.permute(0, 2, 1)  # (8, 48, 301)
        x = self.linear(x)  # (8, 48, 324)
        x = x.permute(0, 2, 1)  # (8, 324, 48)


        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
        ee = x
        weights = ee.detach()
        weights = weights @ weights.transpose(-2, -1)
        weights = self.linear2(weights)
        weights = weights.permute(0, 1, 3, 2)
        weights = self.linear3(weights)
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = x.permute(0, 2, 1)  # (8, 48, 301)
        x = self.linear1(x)  # (8, 48, 324)
        x = x.permute(0, 2, 1)

        return x, weights

# def scaled_dot_product_attention(q, k, v, mask=None):
#     d_k = q.size(-1)  # dimension of key
#     scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     attn_weights = F.softmax(scores, dim=-1)
#     weights = attn_weights
#     output = torch.matmul(attn_weights, v)
#     return output, weights

class XCA(nn.Module):
    #fused_attn: torch.jit.Final[bool]
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        #self.fused_attn = use_fused_attn(experimental=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Result of next line is (qkv, B, num (H)eads,  (C')hannels per head, N)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)


        q = torch.nn.functional.normalize(q, dim=-1) * self.temperature
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))* self.temperature
        attn = attn.softmax(dim= -1)
        #weights = attn.detach()
        attn = self.attn_drop(attn)
        h = (attn @ v).permute(0, 3, 1, 2)
       # weights = h.detach()
       # weights = weights.permute(0, 2, 1, 3)
        #weights = weights @ weights.transpose(-2, -1)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        #weights = x.detach()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x#, weights



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        #self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             # attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = Attention1(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, local_window_size=16)
        #self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0, proj_drop=0)
        #self.attn = LinearAttention(dim=dim, input_resolution=[12, 12], num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.kan = KAN([dim, 24, dim])  # 48 24
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

        self.conv1 = nn.Conv1d(in_channels=48, out_channels=128, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=48, kernel_size=3, padding=1)
        self.sc = nn.Parameter(torch.tensor(0.1))
        self.drop_path1 = DropPath(0.15)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0)
        #self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0, proj_drop=0)
        #self.attention_combiner = AttentionCombiner(input_dim=48, output_dim=48, heads=8)



    def forward(self, x):
        #x = x + self.drop_path(self.attn(self.norm1(x)))
        h = x
        #hh = self.conv_layer(self.norm1(h))
        b, t, d = x.shape
        #b, t, d = x.shape
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        hhh, weights = self.attn(self.norm1(x))

        #hhh1, weights1 = self.attn1(self.norm1(x))
        x = x + self.drop_path(hhh)
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.kan(self.norm2(x).reshape(-1, x.shape[-1])).reshape(b, t, d))


        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = self.drop_path1(h)
        #h = self.conv2(h)
        #h = self.drop_path1(h)
        h = self.conv3(h)
        h = self.drop_path1(h)
        h = h.permute(0, 2, 1)
        h = self.norm1(h)
        #h = self.xca(h)
        #
        #x = self.attention_combiner(x, h)
        x = self.sc * h + x
        #xx = x.permute(0, 2, 1)
        #xx = self.conv1d(xx)
        #xx = self.drop_path1(xx)
        #xx = self.relu(xx)
        #x = xx.permute(0, 2, 1)
        #x = self.attention_combiner(x, xx)

        #x = x + xx
        return x, weights


class AttentionCombiner(nn.Module):
    def __init__(self, input_dim, output_dim, heads=8):
        super(AttentionCombiner, self).__init__()
        self.heads = heads
        self.head_dim = input_dim*2 // heads

        #assert (
                #self.head_dim * heads == input_dim
        #), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, input_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, input_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, input_dim, bias=False)
        self.fc_out = nn.Linear(2*input_dim, output_dim)

    def forward(self, output1, output2):
        N, seq_len, _ = output1.shape

        combined = torch.cat((output1, output2), dim=2)

        values = combined.reshape(N, seq_len, self.heads, self.head_dim)
        keys = combined.reshape(N, seq_len, self.heads, self.head_dim)
        queries = combined.reshape(N, seq_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, seq_len, self.head_dim * self.heads
        )

        out = self.fc_out(out)
        return out

def get_weight(att_mat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    #batch_size, num_heads, seq_len, _ = att_mat.size()
    #residual_att = torch.eye(seq_len).to(device).expand(batch_size, num_heads, seq_len, seq_len)
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v

class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask, fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FeatureEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData) 
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate 
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        #self.blocks = nn.Sequential(*[
        #    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #          norm_layer=norm_layer, act_layer=act_layer)
        #    for i in range(depth)
        #])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        #self.head = KAN([self.num_features, 48,  num_classes]) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)



    def forward_features(self, x):
        h = x
        x = self.feature_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None: #ViT中就是None
            x = torch.cat((cls_token, x), dim=1) 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]),attn_weights 
        else:
            return x[:, 0], x[:, 1],attn_weights

    def forward(self, x):
        #h = x

        latent, attn_weights = self.forward_features(x)

        if self.head_dist is not None: 
            latent, latent_dist = self.head(latent[0]), self.head_dist(latent[1])
            if self.training and not torch.jit.is_scripting():
                return latent, latent_dist
            else:
                return (latent+latent_dist) / 2
        else:
            pre = self.head(latent) 
        return latent, pre, attn_weights

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  

def scTrans_model(num_classes, num_genes, mask, embed_dim=48,depth=2,num_heads=4,has_logits: bool = True):
    model = Transformer(num_classes=num_classes, 
                        num_genes=num_genes, 
                        mask = mask,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
                        representation_size=embed_dim if has_logits else None)
    return model

