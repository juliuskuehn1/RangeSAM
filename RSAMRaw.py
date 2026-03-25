import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from utils.RSAMUtils import *

class RSAMEncoder(nn.Module):
    def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=True, adapter=True, msca=True) -> None:
        super(RSAMEncoder, self).__init__()
        self.backbone = build_backbone(model_cfg)
        self.stem = self.backbone.patch_embed.proj
        out_channels = self.backbone.patch_embed.proj.out_channels
        del self.backbone.patch_embed.proj
        self.msca = msca
        self.stem_config = stem
        if self.stem_config == "pretrained":
            old_conv = self.stem
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None)
            )
            with torch.no_grad():
                new_conv.weight[:, :6, :, :] = torch.cat([old_conv.weight, old_conv.weight], dim=1)
                if new_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.stem = new_conv
        elif self.stem_config == "rangeformer":
            self.stem = nn.Sequential(
                nn.Conv2d(6, 12, kernel_size=1),
                nn.BatchNorm2d(12),
                nn.GELU(),
                nn.Conv2d(12, 24, kernel_size=1),
                nn.BatchNorm2d(24),
                nn.GELU(),
                nn.Conv2d(24, 48, kernel_size=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.Conv2d(48, 96, kernel_size=1),
                nn.BatchNorm2d(96),
                nn.GELU()
            )
        elif self.stem_config == "normal":
            self.stem = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=1),
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
                nn.Conv2d(128, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.GELU(),
                self.stem
            )
        elif self.stem_config == "convnext":
            self.stem = nn.Sequential(
                nn.Conv2d(6, 6, 7, stride=2, padding=3, groups=6),   # (B,6,H/2,W/2)
                nn.Conv2d(6, 96//2, 1),   nn.GELU(),                  # (B,48,H/2,W/2)

                # ── block 1 (pseudo‑LayerNorm) ───────────
                nn.Conv2d(96//2, 96//2, 1),  nn.GELU(),            # still (B,48,H/2,W/2)

                # ── block 2 ───────────────────────────────
                nn.Conv2d(96//2, 96//2, 3, stride=2, padding=1,
                        groups=96//2),                                   # (B,48,H/4,W/4)
                nn.Conv2d(96//2, 96, 1),     nn.GELU(),            # (B,96,H/4,W/4)
            )
        elif self.stem_config == "naive":
            self.stem = nn.Sequential(
                nn.Conv2d(6, 3, 1),
                self.stem
            )
        # if freeze_weight:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        msca_blocks = []
        for dim in [out_channels, out_channels*2, out_channels*4, out_channels*8]:
            if self.msca:
                msca_blocks.append(SpatialAttention(dim))
        blocks = []
        for block in self.backbone.blocks:
            if adapter:
                blocks.append(
                    Adapter(block)
                )
            else:
                blocks.append(block)
        self.backbone.blocks = nn.Sequential(
            *blocks
        )
        self.msca_blocks = nn.ModuleList(msca_blocks)

    def forward(self, x):
        x = self.stem(x)
        if self.stem_config is not None:
            x = x.permute(0, 2, 3, 1)
        x = x + self.backbone._get_pos_embed(x.shape[1:3])
        outputs = []
        stage_idx = 0
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.backbone.stage_ends:
                feat = x.permute(0, 3, 1, 2)
                if self.msca:
                    feat = self.msca_blocks[stage_idx](feat)
                outputs.append(feat)
                x = feat.permute(0, 2, 3, 1)
                stage_idx += 1
        return outputs
    

class RSAMDecoder(nn.Module):
    def __init__(self, out_channels, unify_dim=256, use_rfb=True) -> None:
        super(RSAMDecoder, self).__init__()
        self.use_rfb = use_rfb
        if self.use_rfb:
            self.rfb1 = RFB_modified(out_channels, unify_dim)
            self.rfb2 = RFB_modified(out_channels*2, unify_dim)
            self.rfb3 = RFB_modified(out_channels*4, unify_dim)
            self.rfb4 = RFB_modified(out_channels*8, unify_dim)
        else:
            self.conv1 = nn.Conv2d(out_channels, unify_dim, kernel_size=1)
            self.conv2 = nn.Conv2d(out_channels*2, unify_dim, kernel_size=1)
            self.conv3 = nn.Conv2d(out_channels*4, unify_dim, kernel_size=1)
            self.conv4 = nn.Conv2d(out_channels*8, unify_dim, kernel_size=1)
        self.main_head = nn.Sequential(
            nn.Conv2d(unify_dim * 4, unify_dim * 3, 1),
            nn.GELU(),
            nn.Conv2d(unify_dim * 3, unify_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(unify_dim * 2, unify_dim * 1, 1),
            nn.GELU(),
            nn.Conv2d(unify_dim * 1, 20, 1)
        )
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(unify_dim, 20, 1) for _ in range(4)
        ])

    def forward(self, x):
        x1, x2, x3, x4 = x
        H = 64
        W = 2048
        if self.use_rfb:
            x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        else:
            x1, x2, x3, x4 = self.conv1(x1), self.conv2(x2), self.conv3(x3), self.conv4(x4)
        if x1.shape[-2:] != (H, W):
            x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=(H, W), mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=(H, W), mode='bilinear', align_corners=False)
        pred_main = self.main_head(torch.cat([x1, x2, x3, x4], dim=1))
        pred_aux = [head(h) for head, h in zip(self.aux_heads, [x1, x2, x3, x4])]
        return pred_main, pred_aux  
    

class SAM2UNet(nn.Module):
    def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=False, adapter=False, msca=False, unify_dim=256, use_rfb=True) -> None:
        super(SAM2UNet, self).__init__()
        if any(x in model_cfg for x in ["_s", "_t"]):
            out_channels = 96
        elif any(x in model_cfg for x in ["_l", "_b+"]):
            out_channels = 144
        
        self.encoder = RSAMEncoder(model_cfg=model_cfg, checkpoint_path=checkpoint_path, stem=stem,
                                freeze_weight=freeze_weight, adapter=adapter, msca=msca)
        self.decoder = RSAMDecoder(out_channels=out_channels, unify_dim=unify_dim, use_rfb=use_rfb)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        out_main, out_aux = self.decoder((x1, x2, x3, x4))

        return out_main, out_aux

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet("sam2.1_hiera_t.yaml").cuda()
        # x = torch.randn(1, 6, 64, 2048).cuda()
        # target = (torch.rand(1, 64, 2048) * 20).long().cuda()
        # out_main, out_aux = model(x)
        # loss = combined_loss(out_main, target) + sum(combined_loss(aux_pred, target) for aux_pred in out_aux)



# class RSAMUNETDecoder(nn.Module):
#     def __init__(self, out_channels, unify_dim=256) -> None:
#         super(RSAMUNETDecoder, self).__init__()
#         self.rfb1 = RFB_modified(out_channels, unify_dim)
#         self.rfb2 = RFB_modified(out_channels*2, unify_dim)
#         self.rfb3 = RFB_modified(out_channels*4, unify_dim)
#         self.rfb4 = RFB_modified(out_channels*8, unify_dim)
#         self.up1 = (Up(128, 64))
#         self.up2 = (Up(128, 64))
#         self.up3 = (Up(128, 64))
#         self.up4 = (Up(128, 64))
#         self.side1 = nn.Conv2d(unify_dim, 20, kernel_size=1)
#         self.side2 = nn.Conv2d(unify_dim, 20, kernel_size=1)
#         self.head = nn.Conv2d(unify_dim, 20, kernel_size=1)
        
#         # self.conv1 = nn.Conv2d(out_channels, unify_dim, kernel_size=1)
#         # self.conv2 = nn.Conv2d(out_channels*2, unify_dim, kernel_size=1)
#         # self.conv3 = nn.Conv2d(out_channels*4, unify_dim, kernel_size=1)
#         # self.conv4 = nn.Conv2d(out_channels*8, unify_dim, kernel_size=1)
#         # self.main_head = nn.Sequential(
#         #     nn.Conv2d(unify_dim * 4, unify_dim * 3, 1),
#         #     nn.GELU(),
#         #     nn.Conv2d(unify_dim * 3, unify_dim * 2, 1),
#         #     nn.GELU(),
#         #     nn.Conv2d(unify_dim * 2, unify_dim * 1, 1),
#         #     nn.GELU(),
#         #     nn.Conv2d(unify_dim * 1, 20, 1),
#         #     nn.GELU(),
#         # )
#         # self.aux_heads = nn.ModuleList([
#         #     nn.Conv2d(unify_dim, 20, 1) for _ in range(4)
#         # ])
        
#     def forward(self, x):
#         x1, x2, x3, x4 = x
#         x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
#         # x1, x2, x3, x4 = self.conv1(x1), self.conv2(x2), self.conv3(x3),self.conv4(x4)
#         x = self.up1(x4, x3)
#         out1 = F.interpolate(self.side1(x), scale_factor=4, mode='bilinear')
#         x = self.up2(x, x2)
#         out2 = F.interpolate(self.side2(x), scale_factor=2, mode='bilinear')
#         x = self.up3(x, x1)
#         out = F.interpolate(self.head(x), scale_factor=1, mode='bilinear')
#         out3 = F.interpolate(x4, scale_factor=8, mode='bilinear')
#         return out, torch.concat([out1, out2, out3], dim=1)

# class RSAM3CEncoder(nn.Module):
#     def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=True, adapter=True, msca=True) -> None:
#         super(RSAM3CEncoder, self).__init__()
#         if checkpoint_path is not None:
#             self.backbone = build_backbone(model_cfg, checkpoint_path)
#         else:
#             self.backbone = build_backbone(model_cfg)
#         out_channels = self.backbone.patch_embed.proj.out_channels
#         if freeze_weight:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
#         self.msca = msca
#         msca_blocks = []
#         for dim in [out_channels, out_channels*2, out_channels*4, out_channels*8]:
#             if self.msca:
#                 msca_blocks.append(SpatialAttention(dim))
#         blocks = []
#         for block in self.backbone.blocks:
#             if adapter:
#                 blocks.append(
#                     Adapter(block)
#                 )
#             else:
#                 blocks.append(block)
#         self.backbone.blocks = nn.Sequential(
#             *blocks
#         )
#         self.msca_blocks = nn.ModuleList(msca_blocks)

#     def forward(self, x):
#         x = self.backbone.patch_embed(x)        
#         x = x + self.backbone._get_pos_embed(x.shape[1:3])
#         outputs = []
#         stage_idx = 0
#         for i, blk in enumerate(self.backbone.blocks):
#             x = blk(x)
#             if i in self.backbone.stage_ends:
#                 feat = x.permute(0, 3, 1, 2)
#                 if self.msca:
#                     feat = self.msca_blocks[stage_idx](feat)
#                 outputs.append(feat)
#                 x = feat.permute(0, 2, 3, 1)
#                 stage_idx += 1
#         return outputs

# docker run -it --gpus=all -v /opt/cache/duc_local/semantickitti/dataset:/workspace/dataset pc3163.igd.fraunhofer.de:4567/rangesam:rfb 
# python train.py --hiera_path sam2.1_hiera_tiny.pt --save_path output --epoch 100 --lr 0.001 --batch_size 4 --weight_decay 0.00025 --freeze_weight True

# class RSAM2UNetDecoder(nn.Module):
#     def __init__(self, out_channels, unify_dim=512) -> None:
#         super(RSAM2UNetDecoder, self).__init__()
#         self.up1 = (Up(unify_dim*2, unify_dim))
#         self.up2 = (Up(unify_dim*2, unify_dim))
#         self.up3 = (Up(unify_dim*2, unify_dim))
#         self.rfb1 = RFB_modified(out_channels, unify_dim)
#         self.rfb2 = RFB_modified(out_channels*2, unify_dim)
#         self.rfb3 = RFB_modified(out_channels*4, unify_dim)
#         self.rfb4 = RFB_modified(out_channels*8, unify_dim)
#         self.side1 = nn.Conv2d(unify_dim, 20, kernel_size=1)
#         self.side2 = nn.Conv2d(unify_dim, 20, kernel_size=1)
#         self.side3 = nn.Conv2d(unify_dim, 20, kernel_size=1)
#         self.main_head = nn.Sequential(
#             nn.Conv2d(unify_dim * 3, unify_dim * 2, 1),
#             nn.GELU(),
#             nn.Conv2d(unify_dim * 2, unify_dim * 1, 1),
#             nn.GELU(),
#             nn.Conv2d(unify_dim * 1, 20, 1),
#             nn.GELU(),
#         )
#         # self.aux_heads = nn.ModuleList([
#         #     nn.Conv2d(unify_dim, 20, 1) for _ in range(4)
#         # ])

#     def forward(self, x):
#         x1, x2, x3, x4 = x
#         H = 64
#         W = 2048
#         x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
#         x = self.up1(x4, x3)
#         h1_up = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         out1 = F.interpolate(self.side1(x), scale_factor=4, mode='bilinear')
#         x = self.up2(x, x2)
#         h2_up = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
#         out2 = F.interpolate(self.side2(x), scale_factor=2, mode='bilinear')
#         h3 = self.up3(x, x1)
#         out3 = self.side3(h3)
#         pred_main = self.main_head(torch.cat([h1_up, h2_up, h3], dim=1))
#         pred_aux = out1, out2, out3
#         return pred_main, pred_aux  