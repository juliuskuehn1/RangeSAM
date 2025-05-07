import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


# from mmcv.ops import CARAFEPack
# -------------------------------
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


# class UpCARAFE(nn.Module):
#     def __init__(self, in_ch, out_ch, scale=2):
#         super().__init__()
#         self.reassemble = CARAFEPack(
#             channels=in_ch // 2,  # after concat we will halve channels
#             scale_factor=scale,   # ×2
#             up_kernel=5,          # receptive field = 5×5
#             encoder_kernel=3,     # kernel that predicts the kernels
#         )
#         self.reduce = nn.Conv2d(in_ch, in_ch // 2, 1)  # channel squeeze
#         self.double_conv = DoubleConv(in_ch // 2, out_ch)
#
#     def forward(self, low, high):
#         low = self.reassemble(low)          # learnable upsampling
#         fused = torch.cat([high, low], 1)
#         fused = self.reduce(fused)          # keep memory in check
#         return self.double_conv(fused)

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


class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None, freeze_encoder: bool = False) -> None:
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2.1_hiera_s.yaml"
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
        self.encoder = model.image_encoder.trunk
        self.stem = nn.Sequential(
            # from 5 channels → 16, with 3×3 conv, keep H×W the same:
            nn.Conv2d(5,  16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # final RGB mapping: 128->3 with context from neighbors
            nn.Conv2d(128,  3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        self.better_stem = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=1),    # (5→24), (64,2048) → (64,2048)
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=1),  # (48,32,1024)
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=1),  # (72,16,512)
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.Conv2d(72, 96, kernel_size=1),    # (96,16,512)
            nn.BatchNorm2d(96),
            nn.Sigmoid()
        )
        if True:
            original_conv = model.patch_embed.proj
            old_weight = original_conv.weight 
            old_bias = original_conv.bias
            # Create new Conv2d with 6 input channels
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=96,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            new_weight = torch.zeros((96, 6, 7, 7))

            new_weight[:, :3, :, :] = old_weight
            new_weight[:, 3:, :, :] = old_weight 
            new_conv.weight = nn.Parameter(new_weight)
            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(old_bias.clone())
            model.patch_embed.proj = new_conv

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        msca_blocks = []
        for dim in [96, 192, 384, 768]:
            msca_blocks.append(SpatialAttention(dim))
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                block
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(96, 64)
        self.rfb2 = RFB_modified(192, 64)
        self.rfb3 = RFB_modified(384, 64)
        self.rfb4 = RFB_modified(768, 64)
        self.msca_blocks = nn.ModuleList(msca_blocks)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.sde1 = nn.Conv2d(64, 20, kernel_size=1)
        self.side2 = nn.Conv2d(64, 20, kernel_size=1)
        self.head = nn.Conv2d(64, 20, kernel_size=1)

    def forward(self, x):
        # 1) patch‐embed → (B, H, W, C)
        # x = self.stem(x)
        x = self.encoder.patch_embed(x)
        # x = self.better_stem(x).permute(0, 2, 3, 1)
        # 2) add pos embed
        x = x + self.encoder._get_pos_embed(x.shape[1:3])
        outputs = []
        stage_idx = 0
        for i, blk in enumerate(self.encoder.blocks):
            # 3) apply each MultiScaleBlock to (B, H, W, C)
            x = blk(x)
            # 4) if this block is the end of a stage, grab a feature-map
            if i in self.encoder.stage_ends:
                # permute to (B, C, H, W)
                feat = x.permute(0, 3, 1, 2)
                # 5) here’s where you attach your SpatialAttention:
                feat = self.msca_blocks[stage_idx](feat)
                outputs.append(feat)
                x = feat.permute(0, 2, 3, 1)
                stage_idx += 1

        x1, x2, x3, x4 = outputs
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        x = torch.randn(1, 5, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)