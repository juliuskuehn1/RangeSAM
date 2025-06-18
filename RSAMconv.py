import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from utils.RSAMUtils import *

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
        self.pos_emb = pos_emb
        del self.backbone.pos_embed
        del self.backbone.pos_embed_window
        self.stem = self.backbone.patch_embed.proj
        out_channels = self.backbone.patch_embed.proj.out_channels
        del self.backbone.patch_embed.proj
        # self.stem = nn.Sequential(
        #     nn.Linear(5, out_channels//4),
        #     nn.LayerNorm(out_channels//4),
        #     nn.GELU(),
        #     nn.Linear(out_channels//4, out_channels//2),
        #     nn.LayerNorm(out_channels//2),
        #     nn.GELU(),
        #     nn.Linear(out_channels//2, out_channels),
        #     nn.LayerNorm(out_channels),
        #     nn.GELU(),
        # ) 
        self.stem = nn.Sequential(
            nn.Conv2d(5, out_channels//4, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels//4),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels//2, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.patch_embed = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if freeze_weight:
            for param in self.backbone.parameters():
                param.requires_grad = False
        adapted_blocks = nn.ModuleList()
        for block in self.backbone.blocks:
            adapted_blocks.append(
                BestAdapter(block)
            )
        self.backbone.blocks = adapted_blocks
        if self.pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, out_channels, 4, 128)
            )
            self.pos_embed_window = nn.Parameter(
                torch.zeros(1, out_channels, 4, 4)
            )
    def _get_pos_embed(self, hw) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed
    
    def forward(self, x):
        outputs = []
        #x = x.permute(0, 2, 3, 1)
        x = self.stem(x)
        #x = x.permute(0, 3, 1, 2)
        x = self.patch_embed(x)
        # outputs.append(x)
        x = x.permute(0, 2, 3, 1)
        if (self.pos_emb):
            x = x + self._get_pos_embed(x.shape[1:3])
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in self.backbone.stage_ends:
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)
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
            nn.Conv2d(unify_dim * 4, unify_dim * 1, 1),
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
            x1, x2, x3, x4 = self.rfb1(x1),  self.rfb2(x2), self.rfb3(x3),  self.rfb4(x4)
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
        self.decoder = RSAMDecoder(out_channels=out_channels, unify_dim=unify_dim, use_rfb=use_rfb)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        out_main, out_aux = self.decoder((x1, x2, x3, x4))
        return out_main, out_aux

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet("sam2.1_hiera_t.yaml").cuda()
