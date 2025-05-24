import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from utils.RSAMUtils import *

class RSAMEncoder(nn.Module):
    def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=True, adapter=True, msca=True) -> None:
        super(RSAMEncoder, self).__init__()
        self.backbone = build_backbone(model_cfg, checkpoint_path)
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
                nn.Conv2d( 5, 48, kernel_size=1),
                nn.GroupNorm(1, 48),
                nn.GELU(),
                nn.Conv2d(48, 96, kernel_size=1),
                nn.GroupNorm(1, 96),
                nn.GELU(),
                nn.Conv2d(96, 96, kernel_size=1),
            )
        elif self.stem_config == "normal":
            self.stem = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=1),
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
        if freeze_weight:
            for param in self.backbone.parameters():
                param.requires_grad = False
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
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels*2, out_channels*2, 3, padding=1),
            nn.Conv2d(out_channels*4, out_channels*4, 3, padding=1),
            nn.Conv2d(out_channels*8, out_channels*8, 3, padding=1),
        ])

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
                    out = self.conv[stage_idx](feat)
                    out = self.dropout(self.msca_blocks[stage_idx](feat)) + out
                    feat = feat + out
                outputs.append(feat)
                x = feat.permute(0, 2, 3, 1)
                stage_idx += 1
        return outputs

class RSAMRFEncoder(nn.Module):
    def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=True, adapter=True, msca=True, pos_emb=True) -> None:
        super(RSAMRFEncoder, self).__init__()
        model = build_sam2(model_cfg, checkpoint_path)
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
        self.pos_emb = pos_emb
        out_channels = self.backbone.patch_embed.proj.out_channels
        del self.backbone.patch_embed.proj
        self.stem = nn.Sequential(
            nn.Conv2d( 5, out_channels//2, kernel_size=1),
            nn.GroupNorm(1, out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
        )
        if freeze_weight:
            for param in self.backbone.parameters():
                param.requires_grad = False
        msca_blocks = []
        for dim in [out_channels, out_channels*2, out_channels*4, out_channels*8]:
            msca_blocks.append(SpatialAttention(dim))
        # blocks = []
        # for block in self.backbone.blocks:
        #     blocks.append(
        #         ImprovedAdapter(block)
        #     )
        # self.backbone.blocks = nn.Sequential(
        #     *blocks
        # )
        adapted_blocks = nn.ModuleList()
        for i, blk in enumerate(self.backbone.blocks):
            stage_id   = stage_id_for_block(i)
            num_convs  = NUM_CONVS_PER_STAGE[stage_id]
            adapted_blocks.append(ImprovedAdapterSep(blk, num_convs))
            
        # swap out the original blocks list
        self.backbone.blocks = adapted_blocks
        self.msca_blocks = nn.ModuleList(msca_blocks)
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv1 = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels*2, out_channels*2, 3, padding=1),
            nn.Conv2d(out_channels*4, out_channels*4, 3, padding=1),
            nn.Conv2d(out_channels*8, out_channels*8, 3, padding=1),
        ])
        self.conv2 = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Conv2d(out_channels*2, out_channels*2, 1),
            nn.Conv2d(out_channels*4, out_channels*4, 1),
            nn.Conv2d(out_channels*8, out_channels*8, 1),
        ])
        self.norm = nn.ModuleList([
            nn.GroupNorm(1, out_channels*1),
            nn.GroupNorm(1, out_channels*2),
            nn.GroupNorm(1, out_channels*4),
            nn.GroupNorm(1, out_channels*8),
        ])
        if (self.pos_emb):
            self.window_spec = [8, 4, 14, 7]
            self.window_pos_embed_bkg_spatial_size = [1, 32]
            self.pos_embed_window = nn.Parameter(
                torch.zeros(1, 96, self.window_spec[0], self.window_spec[0])
            )
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 96, *self.window_pos_embed_bkg_spatial_size)
            )
        
    def _get_pos_embed(self) -> torch.Tensor:
        h, w = 64, 2048
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        # x = x + self.backbone._get_pos_embed(x.shape[1:3])
        if (self.pos_emb):
            x = x + self._get_pos_embed()
        outputs = []
        stage_idx = 0
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.backbone.stage_ends:
                feat = x.permute(0, 3, 1, 2)
                out = self.conv1[stage_idx](feat)
                out = self.norm[stage_idx](out)
                out = self.dropout(self.msca_blocks[stage_idx](feat)) + out
                feat = feat + out
                feat = self.conv2[stage_idx](feat)
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
            nn.GroupNorm(1, unify_dim * 3),
            nn.GELU(),
            nn.Conv2d(unify_dim * 3, unify_dim * 2, 1),
            nn.GroupNorm(1, unify_dim * 2),
            nn.GELU(),
            nn.Conv2d(unify_dim * 2, unify_dim * 1, 1),
            nn.GroupNorm(1, unify_dim * 1),
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
    def __init__(self,model_cfg, checkpoint_path=None, stem="rangeformer", freeze_weight=False, adapter=False, msca=False, unify_dim=256, use_rfb=True, pos_emb=True) -> None:
        super(SAM2UNet, self).__init__()
        if any(x in model_cfg for x in ["_s", "_t"]):
            out_channels = 96
        elif any(x in model_cfg for x in ["_l", "_b+"]):
            out_channels = 144
        
        self.encoder = RSAMRFEncoder(model_cfg=model_cfg, checkpoint_path=checkpoint_path, stem=stem,
                                freeze_weight=freeze_weight, adapter=adapter, msca=msca, pos_emb=pos_emb)
        self.decoder = RSAMDecoder(out_channels = out_channels, unify_dim=unify_dim, use_rfb=use_rfb)
        # self.decoder = SegFormerDecoder(
        # embed_dims=[out_channels,
        #             out_channels*2,
        #             out_channels*4,
        #             out_channels*8],
        # unify_dim=unify_dim,
        # num_classes=20)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        out_main, out_aux = self.decoder((x1, x2, x3, x4))
        return out_main, out_aux

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet("sam2.1_hiera_t.yaml").cuda()
