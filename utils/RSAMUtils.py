import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


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
        self.act = nn.LeakyReLU()
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
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

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
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
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
        self.act     = nn.GELU()

        convs = []
        if (num_convs == 0):
            convs.extend([
                nn.Conv2d(self.dim * 4, self.dim * 4, kernel_size=1, groups=self.dim, bias=True),
                nn.GELU(),
            ])
        for _ in range(num_convs):
            convs.extend([
                nn.Conv2d(self.dim * 4, self.dim * 4,
                          kernel_size=(1, 3), padding=(0, 1),
                          groups=self.dim, bias=True),
                nn.GELU(),
                nn.Conv2d(self.dim * 4, self.dim * 4,
                          kernel_size=(3, 1), padding=(1, 0),
                          groups=self.dim, bias=True),
                nn.GELU(),
            ])
        self.dw_stack = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor
        y = self.act(self.linear1(x))
        y = y.permute(0, 3, 1, 2)
        y = self.dw_stack(y)
        y = y.permute(0, 2, 3, 1)
        y = self.act(self.linear2(y))
        out = self.block(x + y)
        return out

class ImprovedAdapter(nn.Module):
    """
    SegFormer-style Adapter that can sit in front of a MultiScaleBlock.
    Works for any spatial resolution because H and W are taken
    from the input on every forward pass.
    """
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features          # = channel dim C

        self.linear1 = nn.Linear(dim, dim*4)
        # self.conv1 = nn.Conv2d(dim, dim*4, kernel_size=1)
        self.dwconv  = nn.Conv2d(
            dim*4, dim*4,
            kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.linear2 = nn.Linear(dim*4, dim)
        # self.conv2 = nn.Conv2d(dim*4, dim, kernel_size=1)
        self.act = nn.GELU()

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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


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


def print_train_config(model_cfg, checkpoint_path, stem, freeze_weight, adapter, msca):
    print("Training config:")
