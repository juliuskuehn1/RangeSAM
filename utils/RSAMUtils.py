import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv1_1(attn)
        attn_0 = self.conv1_2(attn_0)

        attn_1 = self.conv2_1(attn)
        attn_1 = self.conv2_2(attn_1)

        attn_2 = self.conv3_1(attn)
        attn_2 = self.conv3_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv4(attn)

        return attn * x


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class IAC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = BasicConv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, prev, enc_feat):
        up = F.interpolate(enc_feat, size=prev.shape[2:], mode='bilinear',
                           align_corners=True)
        fused = torch.cat((prev, up), dim=1)
        return self.conv(fused)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.downsample = downsample

        self.conv = BasicConv2d(planes, planes, 3, padding=1)

        self.attn = SpatialAttention(planes)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        out = self.conv(x)
        out = out + self.dropout(self.attn(out))

        out += x

        return out


# --------------------------------


def build_backbone(model_cfg, checkpoint_path=None):
    if checkpoint_path:
        model = build_sam2(model_cfg, checkpoint_path)
    else:
        model = build_sam2(model_cfg)
    del model.sam_mask_decoder
    del model.sam_prompt_encoder
    del model.memory_encoder
    del model.memory_attention
    del model.mask_downsample
    del model.obj_ptr_tpos_proj
    del model.obj_ptr_proj
    del model.image_encoder.neck
    return model.image_encoder.trunk

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels *4)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    
def stage_id_for_block(idx: int) -> int:
    # cumulative lengths of stages
    # stage 0: 0               (1 block)          -> indices 0
    # stage 1: 1 ..  2         (2 blocks)         -> indices 1-2
    # stage 2: 3 ..  9         (7 blocks)         -> indices 3-9
    # stage 3: 10 .. 11        (2 blocks)         -> indices 10-11
    if   idx == 0:                    return 0   # stage-1
    elif 1 <= idx <= 2:               return 1   # stage-2
    elif 3 <= idx <= 9:               return 2   # stage-3
    elif 10 <= idx <= 11:             return 3   # stage-4
    else:
        raise ValueError(f"unexpected block index {idx}")

# mapping stage-id ➜ number of 3×3 convolutions
NUM_CONVS_PER_STAGE = {0: 0, 1: 1, 2: 2, 3: 3}

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class ImprovedAdapterSep(nn.Module):
    """
    Same idea, but each 3×3 depth-wise conv is factorised
    into a depth-wise 1×3 followed by 3×1.
    """
    def __init__(self, blk: nn.Module, num_convs: int = 1):
        super().__init__()
        self.block = blk
        self.dim   = blk.attn.qkv.in_features
        self.linear1 = nn.Linear(self.dim, self.dim * 4)
        self.linear2 = nn.Linear(self.dim * 4, self.dim)
        self.norm1 = nn.LayerNorm(self.dim*4)
        self.norm2 = nn.LayerNorm(self.dim)
        self.act     = nn.GELU()

        convs = []
        if (num_convs == 0):
            convs.extend([
                nn.Conv2d(self.dim * 4, self.dim * 4, kernel_size=1, groups=self.dim, bias=True),
                Permute(0, 2, 3, 1),
                nn.LayerNorm(self.dim * 4),
                nn.GELU(),
                Permute(0, 3, 1, 2)
            ])
        for _ in range(num_convs):
            convs.extend([
                nn.Conv2d(self.dim * 4, self.dim * 4,
                          kernel_size=3, padding=1,
                          groups=self.dim, bias=True),
                Permute(0, 2, 3, 1),
                nn.LayerNorm(self.dim * 4),
                nn.GELU(),
                Permute(0, 3, 1, 2)
            ])
        self.dw_stack = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = y.permute(0, 3, 1, 2)
        y = self.dw_stack(y)
        y = y.permute(0, 2, 3, 1)
        y = self.linear2(y)
        y = self.norm2(y)
        y = self.act(y)
        out = self.block(x + y)
        return out

class ImprovedAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features          # = channel dim C

        self.linear1 = nn.Linear(dim, dim*4)
        self.norm1 = nn.LayerNorm(dim*4)
        self.norm2 = nn.LayerNorm(dim*4)
        self.norm3 = nn.LayerNorm(dim)
        self.dwconv  = nn.Conv2d(
            dim*4, dim*4,
            kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.linear2 = nn.Linear(dim*4, dim)
        # self.conv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)                     # (B, H, W, C)
        y = self.norm1(y)
        y = self.act(y)
        y = y.permute(0, 3, 1, 2)               # (B, C, H, W)
        #y = self.conv1(y)
        y = self.dwconv(y)                      # (B, C, H, W)
        y = y.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = self.act(y)
        #y = self.conv2(y)
        y = self.linear2(y)
        y = self.norm3(y)
        y = self.act(y)
        x = x + y                               
        out = self.block(x)                     
        return out

class BestAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.dim_out         
        self.dwconv  = nn.Conv2d(
            dim, dim*4,
            kernel_size=(3,3), padding="same", groups=dim, bias=False
        )
        self.linear = nn.Linear(dim*4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        x = self.block(x)
        out = x.permute(0, 3, 1, 2)
        out = self.act(out)
        out = self.dwconv(out)
        out = out.permute(0, 2, 3, 1)
        out = self.act(out)                                    
        return self.linear(out) + x

class SimplestAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.dim_out
        # self.dwconv  = nn.Conv2d(
        #     dim, dim*4,
        #     kernel_size=3, padding=1, groups=dim, bias=True
        # )
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        x = self.block(x)
        out = self.act(x)
        out = self.linear1(out)
        out = self.act(out)                     
        return self.linear2(out) + x

class VeryBestAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.dim_out          # = channel dim C
        self.act = nn.GELU()
        self.msca_blocks = SpatialAttention(dim*2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.linear1 = nn.Linear(dim, dim*2)
        self.linear2 = nn.Linear(dim*2, dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        feat = self.block(x)
        feat = self.linear1(feat)
        out = self.act(feat)
        out = out.permute(0, 3, 1, 2)
        out = self.dropout(self.msca_blocks(out)) + out
        out = out.permute(0, 2, 3, 1)
        feat = feat + out
        feat = self.linear2(feat)
        return feat
    
    
class UltimateAdapter(nn.Module):
    def __init__(self, blk: nn.Module, num_convs: int = 1):
        super().__init__()
        self.block = blk
        self.dim   = blk.dim_out
        self.linear = nn.Linear(self.dim, self.dim)
        convs = []
        convs.extend([
            nn.Conv2d(self.dim * 1, self.dim * 1, kernel_size=1, groups=self.dim, bias=True),
            nn.GELU(),
        ])
        for _ in range(num_convs):
            convs.extend([
                nn.Conv2d(self.dim * 4, self.dim * 4,
                          kernel_size=3, padding=1,
                          groups=self.dim, bias=True),
                nn.GELU(),
            ])
        self.dw_stack = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        out = x.permute(0, 3, 1, 2)
        out = self.dw_stack(out)
        out = out.permute(0, 2, 3, 1)
        return self.linear(out) + x
    
class DropoutAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features          # = channel dim C
        self.linear1 = nn.Linear(dim, dim*4)
        # self.conv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.dwconv  = nn.Conv2d(
            dim*4, dim*4,
            kernel_size=3, padding=1, groups=dim)
        self.linear2 = nn.Linear(dim*4, dim)
        # self.conv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(p=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)                     # (B, H, W, C)
        y = self.act(y)
        y = y.permute(0, 3, 1, 2)               # (B, C, H, W)
        #y = self.conv1(y)
        y = self.dwconv(y)                      # (B, C, H, W)
        y = self.act(y)
        #y = self.conv2(y)
        y = y.permute(0, 2, 3, 1)               # (B, H, W, C)
        y = self.linear2(y)
        y = self.act(y)
        x = x + y            
        out = self.block(x)                     
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, gelu=False, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        # self.ln = nn.LayerNorm(out_planes)
        self.norm = nn.GroupNorm(num_groups=out_planes//16, num_channels=out_planes)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        #x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x#.permute(0, 3, 1, 2)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class RSegRFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RSegRFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel*1 , out_channel*1, 1, groups=96),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel*1 , out_channel*1, 1, groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(1, 5), padding=(0, 2), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(3, 1), padding=(1, 0), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, 3, padding=3, dilation=3, groups=96)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel*1 , out_channel*1, 1, groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(1, 7), padding=(0, 3), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(5, 1), padding=(2, 0), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, 3, padding=5, dilation=5, groups=96)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel*1 , out_channel*1, 1, groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(1, 9), padding=(0, 4), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, kernel_size=(7, 1), padding=(3, 0), groups=96),
            BasicConv2d(out_channel*1, out_channel*1, 3, padding=7, dilation=7, groups=96)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1, groups=96)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, groups=96)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

def print_train_config(model_cfg, checkpoint_path, stem, freeze_weight, adapter, msca):
    print("Training config:")

############################################

class ConvStem(nn.Module):
    def __init__(self,
                 in_channels=6,
                 base_channels=12,
                 img_size=(64, 2048),
                 patch_stride=(2, 8),
                 embed_dim=96,
                 flatten=False,
                 hidden_dim=192):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.base_channels = base_channels
        self.dropout_ratio = 0.2

        # Build stem, similar to the design in https://github.com/TiagoCortinhal/SalsaNext
        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels, base_channels),
            ResContextBlock(base_channels, base_channels*2),
            ResContextBlock(base_channels*2, base_channels*4),
            ResBlock(base_channels*4, hidden_dim, self.dropout_ratio, pooling=False, drop_out=False))

        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0
        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding = (patch_stride[0] // 2, patch_stride[1] // 2)
        self.proj_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
             nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.patch_stride = patch_stride
        self.patch_size = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    # def get_grid_size(self, H, W):
    #     return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]
        x_base = self.conv_block(x) # B, hidden_dim, image_size[0], image_size[1]
        x = self.proj_block(x_base)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x 


class ResContextBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.LayerNorm(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.LayerNorm(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA1 = self.bn1(resA)
        resA1 = resA1.permute(0, 3, 1, 2)
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA2 = self.bn2(resA)
        resA2 = resA2.permute(0, 3, 1, 2)
        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.LayerNorm(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.LayerNorm(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.LayerNorm(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.LayerNorm(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA1 = self.bn1(resA)
        resA1 = resA1.permute(0, 3, 1, 2)
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA2 = self.bn2(resA)
        resA2 = resA2.permute(0, 3, 1, 2)
        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA3 = self.bn3(resA)
        resA3 = resA3.permute(0, 3, 1, 2)
        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = resA.permute(0, 2, 3, 1)
        resA = self.bn4(resA)
        resA = resA.permute(0, 3, 1, 2)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB
