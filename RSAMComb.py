import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from utils.RSAMUtils import *


class ResNet_34(nn.Module):
    def __init__(self, nclasses=20, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        model = build_sam2("sam2.1_hiera_t.yaml","sam2.1_hiera_tiny.pt")
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.backbone = model.image_encoder.trunk
        del self.backbone.pos_embed
        del self.backbone.pos_embed_window
        self.stem = self.backbone.patch_embed.proj
        out_channels = 96
        del self.backbone.patch_embed.proj
        self.stem = nn.Sequential(
            nn.Conv2d( 5, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels//2, kernel_size=1),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(out_channels//1),
            nn.GELU(),      
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        # adapted_blocks = nn.ModuleList()
        # for mmblk in self.backbone.blocks:
        #     adapted_blocks.append(
        #         DropoutAdapter(mmblk)
        #     )
        # self.backbone.blocks = adapted_blocks
        self.rfb  = RFB_modified(out_channels, 128)
        self.rfb1 = RFB_modified(out_channels, 128)
        self.rfb2 = RFB_modified(out_channels*2, 128)
        self.rfb3 = RFB_modified(out_channels*4, 128)
        self.rfb4 = RFB_modified(out_channels*8, 128)
        # self.rfb  = nn.Conv2d(out_channels*1, 128, kernel_size=1)
        # self.rfb1 = nn.Conv2d(out_channels*1, 128, kernel_size=1)
        # self.rfb2 = nn.Conv2d(out_channels*2, 128, kernel_size=1)
        # self.rfb3 = nn.Conv2d(out_channels*4, 128, kernel_size=1)
        # self.rfb4 = nn.Conv2d(out_channels*8, 128, kernel_size=1)
        #######################################################################################
        self.nclasses = nclasses
        # mos modification
        self.input_size = 5

        print("Depth of backbone input = ", self.input_size)
        ###

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = True

        self.groups = groups
        self.base_width = width_per_group

        # self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        # self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        # self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        # self.decoder1 = BasicConv2d(256, 128, 3, padding=1)
        # self.decoder2 = BasicConv2d(256, 128, 3, padding=1)
        # self.decoder3 = BasicConv2d(256, 128, 3, padding=1)
        # self.decoder4 = BasicConv2d(256, 128, 3, padding=1)

        # self.fusion_conv = BasicConv2d(128 * 3, 128, kernel_size=1)
        # self.semantic_output = nn.Conv2d(128, nclasses, 1)

        # if self.aux:
        #     self.aux_head1 = nn.Conv2d(128, nclasses, 1)
        #     self.aux_head2 = nn.Conv2d(128, nclasses, 1)
        #     self.aux_head3 = nn.Conv2d(128, nclasses, 1)
        self.main_head = nn.Sequential(
            nn.Conv2d(128 * 4, 128 * 2, 1),
            nn.BatchNorm2d(128 * 2),
            nn.GELU(),
            nn.Conv2d(128 * 2, 128 * 1, 1),
            nn.BatchNorm2d(128 * 1),
            nn.GELU(),
            nn.Conv2d(128 * 1, 20, 1)
        )
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(128, 20, 1) for _ in range(4)
        ])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2)),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    # conv1x1(self.inplanes, planes * block.expansion, stride)
                    # SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
                    # nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        outputs = []
        # Stage 0: Outputs of stem go through the 0. RFB block to layer1
        x = self.stem(x)
        # temp = self.rfb(x)
        # outputs.append(temp)
        outputs.append(self.layer1(self.rfb(x)))
        ####
        # N x H x W x C
        x = x.permute(0, 2, 3, 1)
        stage_idx = 0
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.backbone.stage_ends:
                feat = x.permute(0, 3, 1, 2)
                if stage_idx == 0:
                    feat = self.layer2(self.rfb1(feat) + outputs[0])
                elif stage_idx == 1:
                    feat = self.layer3(self.rfb2(feat) + outputs[1])
                elif stage_idx == 2:
                    feat = self.layer4(self.rfb3(feat) + outputs[2])
                elif stage_idx == 3:
                    outputs[3] = self.rfb4(feat) + outputs[3]
                if stage_idx != 3:
                    outputs.append(feat)
                stage_idx += 1
        # x_1 = self.layer1(x)  # 1
        # x_2 = self.layer2(x_1)  # 1/2
        # x_3 = self.layer3(x_2)  # 1/4
        # x_4 = self.layer4(x_3)  # 1/8
        x1, x2, x3, x4 = outputs[0], outputs[1], outputs[2], outputs[3]
        H, W = 64, 2048
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(H, W), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(H, W), mode='bilinear', align_corners=False)
        pred_main = self.main_head(torch.cat([x1, x2, x3, x4], dim=1))
        pred_aux = [head(h) for head, h in zip(self.aux_heads, [x1, x2, x3, x4])]
        return pred_main, pred_aux  