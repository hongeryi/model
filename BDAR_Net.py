import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.cpp_extension import load

# ==============================================================================
# 1. CUDA Kernel & Autograd Function (CRITICAL FOR TRAINING)
# ==============================================================================
try:
    T_MAX = 1024
    wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                    verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])
    print("[Info] RWKV CUDA kernel loaded successfully.")
except Exception as e:
    print(f"[Warning] Could not load wkv_cuda. RWKV modules will run in slow CPU mode or fail. Error: {e}")
    wkv_cuda = None

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        
        # 调用 C++ forward
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        
        if half_mode: y = y.half()
        elif bf_mode: y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        
        # 调用 C++ backward
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
        
        if half_mode:
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (None, None, None, gw, gu, gk, gv)

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    return output.flatten(2).transpose(1, 2)

class ChannelMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # Parameter names match official RWKV for weight loading
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd))
        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, patch_resolution=None):
        xx = x 
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr * kv

class FrequencyAdaptiveSpatialMix(nn.Module):
    """
    [Idea 2 Implementation] 
    Based on VRWKV_SpatialMix but adds Frequency-Adaptive Decay.
    """
    def __init__(self, n_embd, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = 0.25

        # Standard RWKV Parameters
        self.spatial_decay = nn.Parameter(torch.zeros(n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, n_embd]) * 0.5)

        attn_sz = n_embd
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)

        self.decay_gen = nn.Sequential(
            nn.Linear(n_embd, n_embd // 4),
            nn.Tanh(),
            nn.Linear(n_embd // 4, n_embd),
            nn.Sigmoid() 
        )
        self.decay_gen[-2].bias.data.fill_(-3.0) 

    def jit_func(self, x, patch_resolution):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xv, xr = x, x, x
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, patch_resolution=None):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, patch_resolution)
        
        # [Idea 2] Dynamic Decay Calculation
        global_ctx = x.mean(dim=1) # (B, C)
        dynamic_mod = self.decay_gen(global_ctx) # (B, C)
        batch_avg_mod = dynamic_mod.mean(dim=0) # (C)
        
        w = self.spatial_decay + batch_avg_mod 
        
        if wkv_cuda is not None:
            x = RUN_CUDA(B, T, C, w, self.spatial_first, k, v)
        else:
            x = k # CPU Fallback
            
        x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x



def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 确保 filters 和 x 在同一个 device
    filters = filters.to(x.device)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    filters = filters.to(x.device)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x



class WFR_Module(nn.Module):
    """
    [Fusion Module]
    Integrates Wavelet Split + Low-Freq RWKV + High-Freq CNN + Cross-Gating
    """
    def __init__(self, in_channels, wt_levels=1, wave_type='db1'):
        super().__init__()
        self.in_channels = in_channels
        
        # 初始化小波滤波器并注册为 Buffer (随模型移动设备)
        wt_f, iwt_f = create_wavelet_filter(wave_type, in_channels, in_channels, torch.float)
        self.register_buffer('wt_filter', wt_f)
        self.register_buffer('iwt_filter', iwt_f)
        
        # Branch A: Low Frequency -> RWKV (Global)
        self.norm_low = nn.LayerNorm(in_channels)
        self.rwkv_low = FrequencyAdaptiveSpatialMix(in_channels)
        self.channel_mix = ChannelMix(in_channels)
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)


        # Branch B: High Frequency -> CNN (Detail)
        # High freq channels = 3 * in_channels
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels * 3, in_channels * 3, 1) # 1x1 fusion
        )

        # [Idea 3] Cross-Frequency Gating
        self.gate_proj = nn.Linear(in_channels, in_channels * 3)
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        
        # 1. Wavelet Split
        w_out = wavelet_transform(x, self.wt_filter)
        x_ll = w_out[:, :, 0, :, :] # (B, C, H/2, W/2)
        x_high = w_out[:, :, 1:4, :, :] # (B, C, 3, H/2, W/2)
        
        # 2. Low Freq -> RWKV
        low_seq = x_ll.flatten(2).transpose(1, 2)
        # --- 第一部分：RWKV Attention 的 Layer Scale ---
        low_seq = self.ln1(low_seq)
        low_seq = low_seq + self.rwkv_low(low_seq, patch_resolution=(H//2, W//2))
        low_seq = self.ln2(low_seq)
        low_seq = low_seq + self.channel_mix(low_seq, patch_resolution=(H//2, W//2))
        low_global = low_seq
        # 3. High Freq -> CNN
        x_high = x_high.reshape(B, C*3, H//2, W//2)
        high_detail = self.conv_high(x_high)
        
        # 4. Cross Gating
        gate_logits = self.gate_proj(low_global) # (B, N, 3C)
        gate = torch.sigmoid(gate_logits)
        gate_map = gate.transpose(1, 2).view(B, 3*C, H//2, W//2)
        gated_detail = high_detail * gate_map
        
        # 5. Reconstruction
        low_out = low_global.transpose(1, 2).view(B, C, H//2, W//2)
        detail_split = gated_detail.view(B, C, 3, H//2, W//2)
        recon_input = torch.cat([low_out.unsqueeze(2), detail_split], dim=2)
        
        out = inverse_wavelet_transform(recon_input, self.iwt_filter)
        out = out[:, :, :H, :W] 
        
        return self.fusion(out) + shortcut


class BiVRWKV_SpatialMix_CUDA(nn.Module):
    def __init__(self, n_embd, drop_prob=0.1):
        super().__init__()
        self.n_embd = n_embd
        
        # 投影层
        attn_sz = n_embd
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.drop = nn.Dropout(drop_prob)

        self.w = nn.Parameter(torch.ones(n_embd) * (-6.0), requires_grad=True) 
        self.u = nn.Parameter(torch.zeros(n_embd), requires_grad=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2) 

        k = self.key(x_seq)
        v = self.value(x_seq)
        r = self.receptance(x_seq)
        
        if wkv_cuda is not None:
            # ==================== 【关键修改 2：安全约束】 ====================
            # 强制 real_w 永远为负数，防止训练飞掉。
            # w 参数本身可以随意变化，但传入 CUDA 的永远是负的衰减值。
            real_w = -torch.exp(self.w)
            
            # 正向扫描 (Left -> Right)
            x_fwd = RUN_CUDA(B, H*W, C, real_w, self.u, k, v)
            
            # 反向扫描 (Right -> Left)
            k_b = torch.flip(k, dims=[1])
            v_b = torch.flip(v, dims=[1])
            x_bwd = RUN_CUDA(B, H*W, C, real_w, self.u, k_b, v_b)
            x_bwd = torch.flip(x_bwd, dims=[1])
            
            r = torch.sigmoid(r)
        else:
            # CPU Fallback (保持不变)
            k_exp = torch.exp(k - k.max(dim=1, keepdim=True)[0])
            r_sig = torch.sigmoid(r)
            kv = k_exp * v
            wkv_f = torch.cumsum(kv, dim=1) / (torch.cumsum(k_exp, dim=1) + 1e-6)
            kv_b = torch.flip(kv, dims=[1])
            k_b = torch.flip(k_exp, dims=[1])
            wkv_b = torch.cumsum(kv_b, dim=1) / (torch.cumsum(k_b, dim=1) + 1e-6)
            wkv_b = torch.flip(wkv_b, dims=[1])
            x_fwd, x_bwd = wkv_f, wkv_b
            r = r_sig

        # 融合正反向
        x_bi = (x_fwd + x_bwd) * 0.5
        x_out = self.key_norm(r * x_bi)
        x_out = self.output(x_out)
        x_out = self.drop(x_out)
        
        return x_out.transpose(1, 2).view(B, C, H, W)

class HBR_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # 路径 A: 上面定义的 CUDA 版双向 RWKV
        self.global_path = BiVRWKV_SpatialMix_CUDA(dim, drop_prob=0.1)
        
        # 路径 B: 保留原来的卷积结构 (空间锚定)
        self.spatial_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.gamma_g = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)
        self.gamma_s = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)

    def forward(self, x):
        feat_global = self.global_path(x)
        feat_spatial = self.spatial_path(x)
        # 加权融合
        return (feat_global * self.gamma_g) + (feat_spatial * self.gamma_s)

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv2d, self).__init__()
        assert in_channels == out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        wt_f, iwt_f = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.register_buffer('wt_filter', wt_f)
        self.register_buffer('iwt_filter', iwt_f)
        
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        if self.stride > 1: self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else: self.do_stride = None

    def forward(self, x):
        x_ll_in_levels, x_h_in_levels, shapes_in_levels = [], [], []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])
        next_x_ll = 0
        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        x = self.base_scale(self.base_conv(x)) + next_x_ll
        if self.do_stride is not None: x = self.do_stride(x)
        return x
        

class CAR_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 简单的特征融合层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, skip_feat, prev_pred):
        # [新增] 自动容错逻辑：如果是元组，取第一个元素
        if isinstance(prev_pred, (tuple, list)):
            prev_pred = prev_pred[0]
            
        # 1. 将上一级预测上采样到当前尺寸
        prev_pred = F.interpolate(prev_pred, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 2. 生成反向注意力权重
        ra_weight = 1 - torch.sigmoid(prev_pred)
        
        feat = skip_feat * ra_weight
        return self.conv(feat)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        # 两个池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享感知层 (Shared MLP)
        # 使用 1x1 卷积代替 Linear，减少 reshape 操作，速度更快
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别过池化 -> MLP
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        # 相加融合 -> Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out) * x  # 直接在这里乘回去，方便下面调用

class MDAR_Block(nn.Module):
    def __init__(self, in_channels, scale_branches=2, min_channel=32, groups=1):
        super(MDAR_Block, self).__init__()
        
        self.scale_branches = scale_branches
        self.min_channel = min_channel
        
        # 1. 多尺度特征提取分支 (Multi-Scale Branches)
        # scale=0: dilation=1 (普通卷积)
        # scale=1: dilation=2 (膨胀卷积)
        self.multi_scale_branches = nn.ModuleList([])
        # ------------------ 【新增 1】 定义通道注意力列表 ------------------
        self.ca_layers = nn.ModuleList([]) 
        # ------------------------------------------------------------------
        for scale_idx in range(scale_branches):
            # 内部降维以节省计算量 (参考源码逻辑)
            inter_channel = in_channels // (2**scale_idx)
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                # ...
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, 
                          padding=1 + scale_idx, dilation=1 + scale_idx, 
                          groups=groups, bias=False),
                nn.BatchNorm2d(in_channels), nn.PReLU(init=0.0), # [修改] 换成可学习激活函数
                
                # 1x1 调整通道
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inter_channel), nn.PReLU(init=0.0)  # [修改] 换成可学习激活函数
            ))
            # ------------------ 【新增 2】 为每个分支添加注意力 ------------------
            # ratio 设为 8 或 16 都可以，8 更稳一点
            self.ca_layers.append(ChannelAttention(inter_channel, ratio=8))
            # ------------------------------------------------------------------

        # 2. 空间注意力与门控参数
        self.conv1_list = nn.ModuleList([]) # 生成 Attention Map
        self.conv2_list = nn.ModuleList([]) # 映射回原通道
        
        # 核心门控参数 (Learnable Alpha/Beta)
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        for scale_idx in range(scale_branches):
            inter_channel = in_channels // (2**scale_idx)
            if inter_channel < self.min_channel: inter_channel = self.min_channel
            
            # 生成 1通道 空间注意力图 (Sigmoid -> 0~1)
            self.conv1_list.append(nn.Sequential(
                nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            ))
            
            # 恢复通道数
            self.conv2_list.append(nn.Sequential(
                nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels), nn.PReLU(init=0.0) # [修改] 换成可学习激活函数
            ))

    def forward(self, x):
        feature_aggregation = 0
        
        for scale_idx in range(self.scale_branches):
            # A. 提取多尺度特征
            feature = self.multi_scale_branches[scale_idx](x)

            # 在生成空间 Spatial Map 之前，先用 Channel Attention 提纯一下特征
            feature = self.ca_layers[scale_idx](feature)
            # --------------------------------------------------------------
            
            # B. 生成空间注意力图
            spatial_map = self.conv1_list[scale_idx](feature)
            
            # C. 门控融合 (核心公式)
            # Feature * (1 - Map) * Alpha + Feature * Map * Beta
            # Alpha控制背景流，Beta控制前景流
            weighted_feat = feature * (1 - spatial_map) * self.alpha_list[scale_idx] + \
                            feature * spatial_map * self.beta_list[scale_idx]
            
            # D. 映射回原尺寸并累加
            refined_feat = self.conv2_list[scale_idx](weighted_feat)
            feature_aggregation += refined_feat

        # 平均化
        feature_aggregation /= self.scale_branches
        
        # 残差连接
        return x + feature_aggregation


# [适配器] MDAR Decoder
# 作用: 替代 dconv_block，负责将 Concat 后的厚特征降维，然后应用 MDAR
class MDAR_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDAR_Decoder, self).__init__()
        
        # 1. 降维融合 (Fusion)
        # 先把 cat(skip, upsample) 的通道数降下来
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        """self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU() # [建议修改] 原来是 LeakyReLU
        )"""
        
        # 2. MDAR 增强 (保持通道数不变)
        self.mssa = MDAR_Block(out_channels)
        
        # 3. 局部精修 (Refine)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        """self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU() # [建议修改] 原来是 LeakyReLU
        )"""

    def forward(self, x):
        x = self.fusion(x)
        x = self.mssa(x) # 应用 MDAR
        x = self.refine(x)
        return x

class BDAR_Net(nn.Module):
    def __init__(self, in_channels, out_channels, wave_level=1, model_size='small'):
        super(BDAR_Net, self).__init__()
        
        # 1. 确认通道数 (4层结构配置)
        num_channels = [48, 72, 144, 240, 480] 
        
        self.in_conv = nn.Conv2d(in_channels, num_channels[0], kernel_size=1)
        
        # 2. Encoder 1-4 (保持不变)
        self.encoder1 = self.conv_block(num_channels[0], num_channels[1], wave_level)
        self.encoder2 = self.conv_block(num_channels[1], num_channels[2], wave_level)
        self.encoder3 = self.rwkv_stage(num_channels[2], num_channels[3])
        self.encoder4 = self.rwkv_stage(num_channels[3], num_channels[4])

        # 输入通道是 480 (num_channels[4])
        self.middle = HBR_Bottleneck(num_channels[4])
        

        # 注意: RA 不需要全局特征，只需要当前层通道数
        self.car4 = CAR_Block(num_channels[4])
        self.car3 = CAR_Block(num_channels[3])
        self.car2 = CAR_Block(num_channels[2])
        self.car1 = CAR_Block(num_channels[1])

        # RA 依赖于"上一级的预测"，所以每一层decoder都要能输出mask
        self.out_head4 = nn.Conv2d(num_channels[3], 1, 1) # decoder4 输出后的通道
        self.out_head3 = nn.Conv2d(num_channels[2], 1, 1)
        self.out_head2 = nn.Conv2d(num_channels[1], 1, 1)
        # final_conv 就是 head1
        
        # 用 MDAR_Decoder 替换原来的 self.dconv_block
        self.decoder4 = MDAR_Decoder(num_channels[4]*2, num_channels[3])
        self.decoder3 = MDAR_Decoder(num_channels[3]*2, num_channels[2])
        self.decoder2 = MDAR_Decoder(num_channels[2]*2, num_channels[1])
        self.decoder1 = MDAR_Decoder(num_channels[1]*2, num_channels[0])
        
        self.final_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)

    # ... (其余方法保持不变) ...

    def conv_block(self, in_channels, out_channels, wave_level):
        return nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_levels=wave_level),
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
    
    def rwkv_stage(self, in_channels, out_channels):
        return nn.Sequential(
            WFR_Module(in_channels), 
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1), 
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    """def midconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU()
        )

    def dconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU()
        )"""

    def forward(self, x):
        x = self.in_conv(x)
        
        # --- Encoder (保持不变) ---
        enc1 = F.leaky_relu(self.encoder1(x))
        enc2 = F.leaky_relu(self.encoder2(enc1))
        enc3 = F.leaky_relu(self.encoder3(enc2))
        enc4 = F.leaky_relu(self.encoder4(enc3))
        
        middle = self.middle(enc4)
        
        # --- Stage 4 (最深层) ---
        # 1. 拼接: Middle + Enc4
        dec4 = self.decoder4(torch.cat([middle, enc4], 1))
        # 2. 上采样: 恢复双线性插值 (MDAR 不改变尺寸，需要插值放大)
        dec4 = F.leaky_relu(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False))
        # 3. 预测: 生成 Stage 4 的 Mask (用于指导下一级 RA)
        pred4 = self.out_head4(dec4) 
        
        # --- Stage 3 ---
        # 1. 反向注意力 (RA): 用 pred4 里的知识去清洗 enc3 (去除背景/噪声)
        s3 = self.car3(enc3, pred4)
        # 2. 拼接 & 解码
        dec3 = self.decoder3(torch.cat([dec4, s3], 1))
        dec3 = F.leaky_relu(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        pred3 = self.out_head3(dec3)
        
        # --- Stage 2 ---
        s2 = self.car2(enc2, pred3) # 用 pred3 清洗 enc2
        dec2 = self.decoder2(torch.cat([dec3, s2], 1))
        dec2 = F.leaky_relu(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))
        pred2 = self.out_head2(dec2)
        
        # --- Stage 1 (最浅层) ---
        s1 = self.car1(enc1, pred2) # 用 pred2 清洗 enc1
        dec1 = self.decoder1(torch.cat([dec2, s1], 1))
        dec1 = F.leaky_relu(F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False))
        
        # 最终输出
        final_out = self.final_conv(dec1)
        
        # 训练时返回多尺度监督，推理时只返回最终结果
        if self.training:
            return final_out, pred2, pred3, pred4
        else:
            return final_out