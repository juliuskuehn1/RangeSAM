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
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2.1_hiera_t.yaml"
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

        # Keep the original encoder
        self.encoder = model.image_encoder.trunk

        # Modify the first conv layer for 5-channel input
        old_conv = self.encoder.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels=5,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.encoder.patch_embed.proj = new_conv

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MSCA blocks for different feature dimensions
        self.msca_blocks = nn.ModuleList([
            SpatialAttention(96),
            SpatialAttention(192),
            SpatialAttention(384),
            SpatialAttention(768)
        ])

        # Create Adapter modules for encoder blocks
        adapt_blocks = []
        for i, block in enumerate(self.encoder.blocks):
            adapt_blocks.append(Adapter(block))
        self.encoder.blocks = nn.ModuleList(adapt_blocks)

        # RFB blocks
        self.rfb1 = RFB_modified(96, 64)
        self.rfb2 = RFB_modified(192, 64)
        self.rfb3 = RFB_modified(384, 64)
        self.rfb4 = RFB_modified(768, 64)

        # Upsample blocks
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.up4 = Up(128, 64)

        # Output heads
        self.side1 = nn.Conv2d(64, 20, kernel_size=1)
        self.side2 = nn.Conv2d(64, 20, kernel_size=1)
        self.head = nn.Conv2d(64, 20, kernel_size=1)

        # Feature reduction convs
        self.conv1 = nn.Conv2d(96, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(192, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(384, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(768, 64, kernel_size=1)

        # Store the indices where feature dimension changes
        # Based on the torchinfo output
        self.stage_indices = [2, 4, 11]  # After these indices, feature dimensions change

    def rearrange_if_needed(self, x):
        """Convert tensor to channel-first format if needed"""
        if x.shape[-1] > 10:  # Heuristic: if last dim is large, it's probably channels
            return x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x

    def forward(self, x):
        # Initial processing
        x = self.encoder.patch_embed(x)

        # We need to keep track of features at each stage
        features = []
        stage_features = []

        # Process through encoder blocks and collect features at stage boundaries
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)

            # Apply MSCA and collect features at stage boundaries
            if i == 1:  # First stage features (96 channels)
                print(x.shape)
                feat = self.rearrange_if_needed(x)
                feat = self.msca_blocks[0](feat)
                stage_features.append(feat)
            elif i == 3:  # Second stage features (192 channels)
                feat = self.rearrange_if_needed(x)
                feat = self.msca_blocks[1](feat)
                stage_features.append(feat)
            elif i == 10:  # Third stage features (384 channels)
                feat = self.rearrange_if_needed(x)
                feat = self.msca_blocks[2](feat)
                stage_features.append(feat)
            elif i == len(self.encoder.blocks) - 1:  # Last stage features (768 channels)
                feat = self.rearrange_if_needed(x)
                feat = self.msca_blocks[3](feat)
                stage_features.append(feat)

        # Now we have features at each stage processed through MSCA blocks
        x1, x2, x3, x4 = stage_features

        # Apply 1x1 convs for feature reduction
        x1, x2, x3, x4 = self.conv1(x1), self.conv2(x2), self.conv3(x3), self.conv4(x4)

        # Decoder path
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