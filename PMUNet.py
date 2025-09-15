"""
## PRISM: PROGRESSIVE RAIN REMOVAL WITH INTEGRATED STATE-SPACE MODELING 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from pdb import set_trace as stx

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    print("pytorch_wavelets not available, using fallback implementation")

##########################################################################
##########################################################################
# Wavelet Transform Components

class DirectWaveletTransform(nn.Module):
    def __init__(self, channels, J=2, wave='haar', mode='zero'):
        super().__init__()
        self.J = J
        self.wave = wave
        self.mode = mode
        self.channels = channels
        
        if WAVELETS_AVAILABLE:
            self.xfm = DWTForward(J=J, mode=mode, wave=wave)
            self.ifm = DWTInverse(mode=mode, wave=wave)
        
        self.low_freq_processor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.high_freq_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channels, channels, 1)
            ) for _ in range(3)  # LH, HL, HH
        ])
        
    def forward(self, x):
        if not WAVELETS_AVAILABLE:
            return x + self.low_freq_processor(x)
            
        Yl, Yh = self.xfm(x)
        
        LL_enhanced = self.low_freq_processor(Yl)
        
        LH_enhanced = self.high_freq_processors[0](Yh[-1][:, :, 0, :, :])
        HL_enhanced = self.high_freq_processors[1](Yh[-1][:, :, 1, :, :])
        HH_enhanced = self.high_freq_processors[2](Yh[-1][:, :, 2, :, :])
        
        Yh_new = []
        for i, yh in enumerate(Yh):
            if i == len(Yh) - 1:  
                yh_new = torch.stack([LH_enhanced, HL_enhanced, HH_enhanced], dim=2)
                Yh_new.append(yh_new)
            else:
                Yh_new.append(yh)
        
        reconstructed = self.ifm((LL_enhanced, Yh_new))
        return reconstructed + x

##########################################################################

class WaveletPreprocess(nn.Module):
    def __init__(self, in_channels=3, J=2, wave='haar', mode='zero'):
        super().__init__()
        self.wavelet_transform = DirectWaveletTransform(in_channels, J, wave, mode)
        
    def forward(self, img):
        return self.wavelet_transform(img)
    
##########################################################################
## StatisticalSpatialAttention Model 
class StatisticalSpatialAttention(nn.Module):
    def __init__(self, n_feat, kernel_size=7, bias=False):
        super(StatisticalSpatialAttention, self).__init__()
        
        # SSA
        self.feature_conv = conv(n_feat, n_feat, 3, bias=bias)
        self.img_conv = conv(n_feat, 3, 3, bias=bias)
        
        self.spatial_conv = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, x_img):
        x1 = self.feature_conv(x)
        img = self.img_conv(x) + x_img
        
        avg_out = torch.mean(img, dim=1, keepdim=True)
        max_out, _ = torch.max(img, dim=1, keepdim=True)
        var_out = torch.var(img, dim=1, keepdim=True)
        min_out, _ = torch.min(img, dim=1, keepdim=True)

        spatial_attn = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out, var_out, min_out], dim=1))
        )
        
        x1 = x1 * spatial_attn + x
        
        return x1, img

##########################################################################

class WaveletBranch(nn.Module):
    """Wavelet branch - provide frequency domain enhanced features for Stage2's HDM, using SiLU gate mechanism"""
    def __init__(self, dim, J=2, wave='haar', mode='zero'):
        super().__init__()
        self.dim = dim
        self.J = J
        
        if WAVELETS_AVAILABLE:
            self.xfm = DWTForward(J=J, mode=mode, wave=wave)
            self.ifm = DWTInverse(mode=mode, wave=wave)
        
        # Input projection and gating
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        
        # Selective Scan
        self.wavelet_selective_scan = Selective_Scan(
            d_model=dim,
            d_state=8,
            expand=1  # Single-direction scanning
        )
        
        # Output layer
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, x_size, prompt=None):
        if not WAVELETS_AVAILABLE:
            return x
            
        B, L, C = x.shape
        H, W = x_size
        
        # Input projection and split
        xz = self.in_proj(x)
        x_proc, z = xz.chunk(2, dim=-1)  # Split for gating
        
        # Convert to image format
        prepare = x_proc.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # DWT decomposition
        Yl, Yh = self.xfm(prepare)
        
        # h00 reorder
        h00 = torch.zeros_like(prepare)
        for i in range(len(Yh)):
            if i == len(Yh) - 1:  # Highest layer
                h00[:, :, :Yl.size(2), :Yl.size(3)] = Yl
                h00[:, :, :Yl.size(2), Yl.size(3):Yl.size(3)*2] = Yh[i][:, :, 0, :, :]
                h00[:, :, Yl.size(2):Yl.size(2)*2, :Yl.size(3)] = Yh[i][:, :, 1, :, :]
                h00[:, :, Yl.size(2):Yl.size(2)*2, Yl.size(3):Yl.size(3)*2] = Yh[i][:, :, 2, :, :]
        
        # Convert to sequence format
        h00_seq = h00.permute(0, 2, 3, 1).view(B, -1, C)
        
        # Generate default prompt (if not provided)
        if prompt is None:
            prompt = torch.zeros(B, H*W, 8, device=x.device, dtype=x.dtype)
        
        # Selective Scan
        h11_seq = self.wavelet_selective_scan(h00_seq, prompt)
        h11 = h11_seq.permute(0, 2, 1).view(B, C, H, W)
        
        # Restore wavelet coefficients
        for i in range(len(Yh)):
            if i == len(Yh) - 1:
                Yl = h11[:, :, :Yl.size(2), :Yl.size(3)]
                Yh[i][:, :, 0, :, :] = h11[:, :, :Yl.size(2), Yl.size(3):Yl.size(3)*2]
                Yh[i][:, :, 1, :, :] = h11[:, :, Yl.size(2):Yl.size(2)*2, :Yl.size(3)]
                Yh[i][:, :, 2, :, :] = h11[:, :, Yl.size(2):Yl.size(2)*2, Yl.size(3):Yl.size(3)*2]
        
        # IDWT reconstruction
        recons = self.ifm((Yl, Yh))
        y = recons.permute(0, 2, 3, 1).view(B, L, C)
        
        # out_norm → SiLU gating → out_proj
        y = self.out_norm(y)
        y = y * F.silu(z)  # SiLU gating mechanism
        out = self.out_proj(y)
        
        return out + x  # Residual connection

##########################################################################
##########################################################################

# initialization function
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)

            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################

##  HybridAttentiveUNet
class HybridAttentiveEncoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, csff):
        super(HybridAttentiveEncoder, self).__init__()
        
        #  norm1 → CAB → norm2 → WindowAttention → norm3 → ConvFFN → Residual
        
        # Level 1: 40 channels, 2 heads, shift=0
        self.norm1_level1 = nn.LayerNorm(n_feat)
        self.cab1_level1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level1 = nn.LayerNorm(n_feat)
        self.attn_level1 = WindowAttentionAdapter(n_feat, window_size=8, num_heads=2, shift_size=0)
        self.norm3_level1 = nn.LayerNorm(n_feat)
        self.convffn_level1 = ConvFFN(in_features=n_feat, hidden_features=int(n_feat * 2), kernel_size=5)
        
        # Level 2: 80channels, 4heads, shift=4
        self.norm1_level2 = nn.LayerNorm(n_feat * 2)
        self.cab1_level2 = CAB(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level2 = nn.LayerNorm(n_feat * 2)
        self.attn_level2 = WindowAttentionAdapter(n_feat * 2, window_size=8, num_heads=4, shift_size=4)
        self.norm3_level2 = nn.LayerNorm(n_feat * 2)
        self.convffn_level2 = ConvFFN(in_features=n_feat * 2, hidden_features=int(n_feat * 4), kernel_size=5)
        
        # Level 3: 160channels, 8heads, shift=0
        self.norm1_level3 = nn.LayerNorm(n_feat * 4)
        self.cab1_level3 = CAB(n_feat * 4, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level3 = nn.LayerNorm(n_feat * 4)
        self.attn_level3 = WindowAttentionAdapter(n_feat * 4, window_size=8, num_heads=8, shift_size=0)
        self.norm3_level3 = nn.LayerNorm(n_feat * 4)
        self.convffn_level3 = ConvFFN(in_features=n_feat * 4, hidden_features=int(n_feat * 8), kernel_size=5)

        self.down12 = DownSample(n_feat)      # 40 → 80
        self.down23 = DownSample(n_feat * 2)  # 80 → 160

        # Cross Stage Feature Fusion (CSFF) 
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat * 2, n_feat * 2, kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat * 4, n_feat * 4, kernel_size=1, bias=bias)

    def _process_level(self, x, norm1, cab, norm2, attn, norm3, convffn):
        """ norm1 → CAB → norm2 → WindowAttention → norm3 → ConvFFN → Residual"""
        B, C, H, W = x.shape
        shortcut = x
        
        # Convert to sequence format: [B, C, H, W] → [B, H*W, C]
        x_seq = x.permute(0, 2, 3, 1).view(B, H * W, C)
        
        # norm1 → CAB(need to convert back to image format)
        x_norm1 = norm1(x_seq)  # [B, H*W, C]
        x_img = x_norm1.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x_cab = cab(x_img)  # CAB
        
        # norm2 → WindowAttention
        x_seq2 = x_cab.permute(0, 2, 3, 1).view(B, H * W, C)  # Convert back to sequence
        x_norm2 = norm2(x_seq2)
        x_img2 = x_norm2.view(B, H, W, C).permute(0, 3, 1, 2)  # Convert back to image for attention
        x_attn = attn(x_img2)  # WindowAttention
        
        # norm3 → ConvFFN → Residual
        x_seq3 = x_attn.permute(0, 2, 3, 1).view(B, H * W, C)  # Convert back to sequence
        x_norm3 = norm3(x_seq3)
        x_ffn = convffn(x_norm3, (H, W))  # ConvFFN
        
        # Residual connection
        x_seq_shortcut = shortcut.permute(0, 2, 3, 1).view(B, H * W, C)
        x_out_seq = x_ffn + x_seq_shortcut
        
        # Convert back to image format
        x_out = x_out_seq.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x_out

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        # Level 1
        enc1 = self._process_level(x, 
                                   self.norm1_level1, self.cab1_level1, self.norm2_level1,
                                   self.attn_level1, self.norm3_level1, self.convffn_level1)
        
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        # Level 2
        enc2 = self._process_level(x,
                                   self.norm1_level2, self.cab1_level2, self.norm2_level2,
                                   self.attn_level2, self.norm3_level2, self.convffn_level2)
        
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        # Level 3
        enc3 = self._process_level(x,
                                   self.norm1_level3, self.cab1_level3, self.norm2_level3,
                                   self.attn_level3, self.norm3_level3, self.convffn_level3)
        
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class HybridAttentiveDecoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(HybridAttentiveDecoder, self).__init__()
        
        #norm1 → CAB → norm2 → WindowAttention → norm3 → ConvFFN → Residual
        # Cross-shift mechanism: [shift=4, shift=0, shift=4] (Level 3,2,1)
        
        # Level 3: 160channels, 8heads, shift=4 (Cross-shift)
        self.norm1_level3 = nn.LayerNorm(n_feat * 4)
        self.cab1_level3 = CAB(n_feat * 4, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level3 = nn.LayerNorm(n_feat * 4)
        self.attn_level3 = WindowAttentionAdapter(n_feat * 4, window_size=8, num_heads=8, shift_size=4)
        self.norm3_level3 = nn.LayerNorm(n_feat * 4)
        self.convffn_level3 = ConvFFN(in_features=n_feat * 4, hidden_features=int(n_feat * 8), kernel_size=5)
        
        # Level 2: 80channels, 4heads, shift=0 (Cross-shift)
        self.norm1_level2 = nn.LayerNorm(n_feat * 2)
        self.cab1_level2 = CAB(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level2 = nn.LayerNorm(n_feat * 2)
        self.attn_level2 = WindowAttentionAdapter(n_feat * 2, window_size=8, num_heads=4, shift_size=0)
        self.norm3_level2 = nn.LayerNorm(n_feat * 2)
        self.convffn_level2 = ConvFFN(in_features=n_feat * 2, hidden_features=int(n_feat * 4), kernel_size=5)
        
        # Level 1: 40channels, 2heads, shift=4 (Cross-shift)
        self.norm1_level1 = nn.LayerNorm(n_feat)
        self.cab1_level1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.norm2_level1 = nn.LayerNorm(n_feat)
        self.attn_level1 = WindowAttentionAdapter(n_feat, window_size=8, num_heads=2, shift_size=4)
        self.norm3_level1 = nn.LayerNorm(n_feat)
        self.convffn_level1 = ConvFFN(in_features=n_feat, hidden_features=int(n_feat * 2), kernel_size=5)

        # SkipUpSample structure
        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat * 2, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat * 2)  # 80 → 40
        self.up32 = SkipUpSample(n_feat * 4)  # 160 → 80

    def _process_level(self, x, norm1, cab, norm2, attn, norm3, convffn):
        """norm1 → CAB → norm2 → WindowAttention → norm3 → ConvFFN → Residual"""
        B, C, H, W = x.shape
        shortcut = x
        
        # Convert to sequence format: [B, C, H, W] → [B, H*W, C]
        x_seq = x.permute(0, 2, 3, 1).view(B, H * W, C)
        
        # norm1 → CAB(need to convert back to image format)
        x_norm1 = norm1(x_seq)  # [B, H*W, C]
        x_img = x_norm1.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x_cab = cab(x_img)  # CAB
        
        # norm2 → WindowAttention
        x_seq2 = x_cab.permute(0, 2, 3, 1).view(B, H * W, C)  # Convert back to sequence
        x_norm2 = norm2(x_seq2)
        x_img2 = x_norm2.view(B, H, W, C).permute(0, 3, 1, 2)  # Convert back to image for attention
        x_attn = attn(x_img2)  # WindowAttention
        
        # norm3 → ConvFFN → Residual
        x_seq3 = x_attn.permute(0, 2, 3, 1).view(B, H * W, C)  # Convert back to sequence
        x_norm3 = norm3(x_seq3)
        x_ffn = convffn(x_norm3, (H, W))  # ConvFFN
        
        # Residual connection
        x_seq_shortcut = shortcut.permute(0, 2, 3, 1).view(B, H * W, C)
        x_out_seq = x_ffn + x_seq_shortcut
        
        # Convert back to image format
        x_out = x_out_seq.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x_out

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        
        # Level 3 (shift=4)
        dec3 = self._process_level(enc3,
                                   self.norm1_level3, self.cab1_level3, self.norm2_level3,
                                   self.attn_level3, self.norm3_level3, self.convffn_level3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        
        # Level 2 (shift=0)
        dec2 = self._process_level(x,
                                   self.norm1_level2, self.cab1_level2, self.norm2_level2,
                                   self.attn_level2, self.norm3_level2, self.convffn_level2)

        x = self.up21(dec2, self.skip_attn1(enc1))
        
        # Level 1 (shift=4)
        dec1 = self._process_level(x,
                                   self.norm1_level1, self.cab1_level1, self.norm2_level1,
                                   self.attn_level1, self.norm3_level1, self.convffn_level1)

        return [dec1, dec2, dec3]

##########################################################################

# Window attention adapter - adapt WindowAttention to 2D convolution form
class WindowAttentionAdapter(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.scale = (dim // num_heads) ** -0.5
        
        # Relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Generate relative position index
        coords = torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to sequence form for window attention
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Calculate attention mask for shifted windows
        attn_mask = None
        if self.shift_size > 0:
            # Create attention mask for shifted windows
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        # Apply Shift mechanism
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        x_windows = window_partition(x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, window_size^2, C]
        
        # Window attention with mask
        attn_windows = self._window_attention(x_windows, attn_mask)
        
        # Restore original shape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [B*num_windows, window_size, window_size, C]
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]
        
        # Restore Shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x
    
    def _window_attention(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask for shifted windows
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, input_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, 1, 1, 0, bias=False), 
            nn.PixelUnshuffle(2)  # [B, input_channels//2, H, W] → [B, input_channels*2, H/2, W/2]
        )

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, input_channels):
        super(UpSample, self).__init__()
        output_channels = input_channels // 2
        self.up = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * 2, 1, 1, 0, bias=False),  
            nn.PixelShuffle(2)  # [B, input_channels*2, H, W] → [B, output_channels, H*2, W*2]
        )

    def forward(self, x):
        return self.up(x)

class SkipUpSample(nn.Module):
    def __init__(self, input_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * 2, 1, 1, 0, bias=False),  
            nn.PixelShuffle(2) 
        )

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
##########################################################################

##HDM Model
def index_reverse(index):
    """
    Reverse index function. Given an index tensor, return the reversed index.
    Parameters:
        index: A tensor of shape [B, HW], representing the index.
    Return:
        index_r: The reversed index tensor, with the same shape as the input.
    """
    index_r = torch.zeros_like(index)  # Create a zero tensor with the same shape as the input index
    ind = torch.arange(0, index.shape[-1]).to(index.device)  # Generate indices from 0 to HW
    for i in range(index.shape[0]):  # Iterate over each batch
        index_r[i, index[i, :]] = ind  # Assign values based on the index
    return index_r

def semantic_neighbor(x, index):
    """
    Rearrange the tensor based on the index, similar to the operation of neighbors in image convolution.
    Parameters:
        x: Input tensor, shape [B, N, C].
        index: Index tensor, shape [B, N], representing how to rearrange the input tensor.
    Return:
        shuffled_x: The tensor rearranged based on the index, with the same shape as the input.
    """
    dim = index.dim()  # Get the dimension of the index tensor
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) have different shapes".format(x.shape, index.shape)  # Ensure input and index shapes match

    # Adjust the dimension of the index based on the dimension relationship between the input tensor and the index
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)  # Expand the index to match the shape of the input tensor

    # Rearrange the input tensor based on the index
    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x
##########################################################################

class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs only 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y
    
##########################################################################

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # Generate relative position index
        coords = torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    
    # Check whether it can be divided evenly    
    if H % window_size != 0 or W % window_size != 0:
        # Calculate the required fill
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        
        # Perform filling (fill in the H and W dimensions)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    # Calculate the actual filled dimensions
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_padded = H + pad_h
    W_padded = W + pad_w
    
    B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
    
    # Use the filled dimensions to reshape
    x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
    
    # If the original size is smaller than the filled size, crop back to the original size
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :]
    
    return x.contiguous()

##########################################################################

class HybridDomainMamba(nn.Module):
    """HDMamba - Core module for semantic reordering and wavelet branch frequency domain enhancement in Stage2"""
    def __init__(self, dim=40, d_state=8, input_resolution=None, num_tokens=64, inner_rank=32, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.inner_rank = inner_rank
        self.num_tokens = num_tokens
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )
        
        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )
        
        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )
        
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        
        self.prompt_proj = nn.Linear(self.inner_rank, self.d_state)
        
        # Wavelet Branch
        self.wavelet_branch = WaveletBranch(dim)
        
        # Adaptive fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, x_size, token):
        B, L, C = x.shape
        H, W = x_size
        x_input = x.clone()
        
        if hasattr(token, 'weight'):
            full_embedding = self.embeddingB.weight @ token.weight
        else:
            full_embedding = self.embeddingB.weight
        
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_proj = self.in_proj(x_img)
        x_proj = x_proj * torch.sigmoid(self.CPE(x_proj))
        x_seq = x_proj.permute(0, 2, 3, 1).view(B, -1, x_proj.size(1))
        
        pred_route = self.route(x_input)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        
        if hasattr(token, 'weight'):
            prompt = torch.matmul(cls_policy, full_embedding).view(B, L, self.d_state)
        else:
            prompt = torch.matmul(cls_policy, full_embedding)
            prompt = self.prompt_proj(prompt)
        
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, L)
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        semantic_x = semantic_neighbor(x_seq, x_sort_indices)
        
        y_assm = self.selectiveScan(semantic_x, prompt)
        y_assm = self.out_proj(self.out_norm(y_assm))
        x_assm = semantic_neighbor(y_assm, x_sort_indices_reverse) + x_input
        
        # Wavelet Branch - independent processing, no semantic prompt
        x_wavelet = self.wavelet_branch(x, x_size, prompt=None)
        
        # Gate fusion
        concat_feat = torch.cat([x_assm, x_wavelet], dim=-1)
        gate = self.fusion_gate(concat_feat)
        x_fused = gate * x_assm + (1 - gate) * x_wavelet
        
        return x_fused

##########################################################################

class AttentiveLayer(nn.Module):
    def __init__(self, dim, d_state=8, num_tokens=64, inner_rank=32, mlp_ratio=2., 
                 window_size=8, num_heads=4, shift_size=0, is_last=False):
        super(AttentiveLayer, self).__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.is_last = is_last
        self.inner_rank = inner_rank
        
        self.norm1 = nn.LayerNorm(dim)  
        self.hdm = HybridDomainMamba(  # HDM
            dim=dim, 
            d_state=d_state,
            num_tokens=num_tokens, 
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )
        self.norm2 = nn.LayerNorm(dim)  
        self.convffn = ConvFFN(in_features=dim, hidden_features=int(dim * mlp_ratio), kernel_size=5)
        
        self.scale = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)
        
        # embeddingA
        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)
        
        # Only the last layer needs image reconstruction
        if is_last:
            self.img_reconstruction = conv(dim, 3, 3, bias=False)
            #  SAM style spatial attention feedback component
            self.spatial_feedback = conv(3, dim, 3, bias=False)  # Generate spatial attention from image
    
    def forward(self, x, x_size, img=None, params=None):
        # HDM
        B, L, C = x.shape
        shortcut = x
        
        # norm1 → HDM → Residual
        x_hdm = self.hdm(self.norm1(x), x_size, self.embeddingA) + x
        
        # norm2 → ConvFFN → Residual
        x = x_hdm + self.convffn(self.norm2(x_hdm), x_size)
        
        # Scale Residual
        x = shortcut * self.scale + x
        
        # Check if it is the last layer
        if self.is_last and img is not None:
            # Add SAM style feedback mechanism
            x_img = x.transpose(1, 2).view(B, C, *x_size)  # [B, dim, H, W]
            
            # Image reconstruction
            reconstructed_img = self.img_reconstruction(x_img) + img  # [B, 3, H, W]
            
            # Generate spatial attention from reconstructed image
            spatial_attn = torch.sigmoid(self.spatial_feedback(reconstructed_img))  # 3 → dim
            
            # Feature modulation
            x_img_modulated = x_img * spatial_attn + x_img  # 特征 × 注意力 + 残差
            
            return x_img_modulated, reconstructed_img
        else:
            # Middle layer: output sequence format feature
            return x, None

##########################################################################

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size, padding=kernel_size//2, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        # x: [B, H*W, C]
        # Add debug information and error handling
        if len(x.shape) != 3:
            raise ValueError(f"ConvFFN input should be 3D tensor, got shape: {x.shape}")
        
        B, L, C = x.shape
        H, W = x_size
        
        x = self.fc1(x)  # [B, H*W, hidden_features]
        x = self.act(x)
        
        # Convert to image format for depthwise convolution
        x = x.transpose(1, 2).view(B, -1, H, W)  # [B, hidden_features, H, W]
        x = self.dwconv(x)  # Depthwise convolution
        x = x.view(B, -1, L).transpose(1, 2)  # [B, H*W, hidden_features]
        
        x = self.fc2(x)  # [B, H*W, out_features]
        return x

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]#4
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################

class ORSNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORSNet, self).__init__()

        input_channels = n_feat  # 40 channels
        # Enhanced feature dimension: 40 + 20 = 60
        enhanced_feat = n_feat + n_feat // 2  # 40 + 20 = 60
        
        self.orb1 = ORB(enhanced_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(enhanced_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(enhanced_feat, kernel_size, reduction, act, bias, num_cab)

        # Upsampling module: unify multi-scale features to base dimension
        self.up_enc1 = UpSample(n_feat * 2)  # 80 → 40
        self.up_dec1 = UpSample(n_feat * 2)  # 80 → 40

        # Continuous upsampling: 160 → 80 → 40
        self.up_enc2 = nn.Sequential(
            UpSample(n_feat * 4),  # 160 → 80
            UpSample(n_feat * 2)   # 80 → 40
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat * 4),  # 160 → 80
            UpSample(n_feat * 2)   # 80 → 40
        )

        # Feature conversion convolution: 40 → 60
        self.conv_enc1 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, enhanced_feat, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x

##########################################################################

class PMUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(PMUNet, self).__init__()
        # Save parameters as instance variables 
        self.n_feat = n_feat  # 40: base dimension, dimension change: 40 → 80 → 160
        
        # Define activation function
        act = nn.PReLU()
        
        # First stage -  HA-UNet + SSA
        self.wavelet_preprocess = WaveletPreprocess(in_channels=in_c, J=2, wave='haar', mode='zero')
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                          CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = HybridAttentiveEncoder(n_feat, kernel_size, reduction, act, bias, csff=False)
        self.stage1_decoder = HybridAttentiveDecoder(n_feat, kernel_size, reduction, act, bias)
        
        # Stage1 Statistical spatial attention module
        self.stage1_spatial_attn = StatisticalSpatialAttention(n_feat, kernel_size=3, bias=bias)
        
        # Second stage - Hybrid Attentive U-Net
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                          CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage2_encoder = HybridAttentiveEncoder(n_feat, kernel_size, reduction, act, bias, csff=True)
        self.stage2_decoder = HybridAttentiveDecoder(n_feat, kernel_size, reduction, act, bias)

        # Second stage - HA-UNet + ASSM Stage2 Norm layer and ConvFFN after HA-UNet
        self.stage2_norm = nn.LayerNorm(n_feat)  
        self.stage2_convffn = ConvFFN(in_features=n_feat, hidden_features=int(n_feat * 2), kernel_size=5)

        # HDM AttentiveLayer stacking design
        # Define multiple AttentiveLayer stacking, the last layer is marked as is_last=True
        self.attentive_layers = nn.ModuleList([
            AttentiveLayer(
                dim=n_feat, 
                d_state=8, 
                num_tokens=64, 
                inner_rank=32, 
                mlp_ratio=2., 
                is_last=(i == 9)  # The last layer (10th layer) is responsible for image reconstruction
            ) 
            for i in range(10)  # Adjusted to 10 layers of stacking
        ])

        # Third stage - ORSNet
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                          CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage3_orsnet = ORSNet(n_feat, kernel_size, reduction, act, bias, num_cab)

        # Define connection layer
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + n_feat // 2, kernel_size, bias=bias)  # 40*2 -> 60

        # Final output layer
        self.tail = conv(n_feat + n_feat // 2, out_c, kernel_size, bias=bias)  # 60 -> 3

        # Define padding size
        self.padder_size = 2 ** 3

    def check_image_size(self, x):
        _, _, h, w = x.size()  # Get image height and width
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size  # Calculate padding height
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size  # Calculate padding width
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))  # Pad image
        return x
    

    def forward(self, x3_img):
        # 1. Get input size and pad
        B, C, Hi, Wi = x3_img.shape
        x3_img = self.check_image_size(x3_img)
        H = x3_img.size(2)
        W = x3_img.size(3)
        
        # Get n_feat value (from the first convolution layer of shallow_feat1)
        n_feat = self.shallow_feat1[0].out_channels

        ##-------------------------------------------
        ##-------------- Stage 1 (Full image processing)---------
        ##-------------------------------------------
        # Full image HA-UNet processing (directly use original image)
        # wavelet_enhanced_img = self.wavelet_preprocess(x3_img)  
        x1_feat = self.shallow_feat1(x3_img)
        feat1_encoded = self.stage1_encoder(x1_feat)
        feat1_decoded = self.stage1_decoder(feat1_encoded)
        
        # 应用小波统计空间注意力模块
        stage1_feats, stage1_img = self.stage1_spatial_attn(feat1_decoded[0], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 2 (Full image processing)---------
        ##-------------------------------------------
        # Full image processing shallow features
        x2_full = self.shallow_feat2(x3_img)

        # Connect Stage1 features
        x2_input = self.concat12(torch.cat([x2_full, stage1_feats], 1))

        # Full image HA-UNet processing
        feat2_encoded = self.stage2_encoder(x2_input, feat1_encoded, feat1_decoded)
        feat2_decoded = self.stage2_decoder(feat2_encoded)

        # norm and ConvFFN processing after HA-UNet
        feat2_processed = feat2_decoded[0]  # [B, n_feat, H, W]
        feat2_processed = feat2_processed.permute(0, 2, 3, 1)  # [B, H, W, n_feat]
        feat2_processed = self.stage2_norm(feat2_processed)  # norm processing
        feat2_processed = feat2_processed.permute(0, 3, 1, 2)  # [B, n_feat, H, W]
        feat2_processed = feat2_processed.view(B, n_feat, -1).transpose(1, 2)  # [B, H*W, n_feat]
        feat2_processed = self.stage2_convffn(feat2_processed, (H, W))  # ConvFFN processing

        # Prepare HDM processing
        feat2_reshaped = feat2_processed  # [B, H*W, n_feat]
        x_size = (H, W)

        # Use new AttentiveLayer stacking design
        x = feat2_reshaped  # [B, H*W, n_feat] sequence format
        for i, attentive_layer in enumerate(self.attentive_layers):
            if i == len(self.attentive_layers) - 1:
                # The last layer: output image format feature
                x3_samfeats, stage2_img = attentive_layer(x, x_size, img=x3_img, params=None)
            else:
                # Middle layer: output sequence format feature
                x, _ = attentive_layer(x, x_size, params=None)

        ##-------------------------------------------
        ##-------------- Stage 3 (Full image processing)---------
        ##-------------------------------------------
        # Calculate shallow features
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        # Process through ORSNet
        x3_cat = self.stage3_orsnet(x3_cat, feat2_encoded, feat2_decoded)

        # Final output image
        stage3_img = self.tail(x3_cat)

        # Ensure the number of channels of the output image matches the input image
        if stage3_img.size(1) != C:
            stage3_img = stage3_img[:, :C, :, :]
        if stage2_img.size(1) != C:
            stage2_img = stage2_img[:, :C, :, :]
        if stage1_img.size(1) != C:
            stage1_img = stage1_img[:, :C, :, :]

        # Return the output images of the three stages (cropped to the original size)
        return [stage3_img[:, :, :Hi, :Wi] + x3_img[:, :, :Hi, :Wi], stage2_img[:, :, :Hi, :Wi],
                stage1_img[:, :, :Hi, :Wi]]

def main():
    # Create PMUNet model instance
    model = PMUNet()

    # Create a tensor of shape (1, 3, 256, 256), representing a batch of images
    input_tensor = torch.randn(1, 3, 256, 256).cuda()  # Sim
    model = model.cuda()

    # Model forward inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Output results
    for i, output in enumerate(outputs):
        print(f"Stage {i + 1} Output Shape: {output.shape}")
        # You can visualize or save the output of each stage, if needed:
        # For example, save as an image
        # utils.save_image(output, f"output_stage_{i + 1}.png")

if __name__ == '__main__':
    main()
