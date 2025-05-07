import torch

from SAM2UNet import SAM2UNet


def hook_fn(name):
    def _hook(module, inp, out):
        # out might be a tuple (q,k,v) or a tensor; adjust as neededafrom torchinfo import summary
        # summary(model.encoder,
        #         input_size=(1,5,352,352),
        #         col_names=["input_size","output_size"])
        feat = out[0] if isinstance(out, (list,tuple)) else out
        # print(f"{name:15} -> {feat.shape}")
    return _hook

model = SAM2UNet().cuda()
for idx, blk in enumerate(model.encoder.blocks):
    blk.register_forward_hook(hook_fn(f"block{idx:02d}"))

# now run a pass
x = torch.randn(4,5,128,4096).cuda()
with torch.no_grad():
    _ = model.encoder(x)


from torchinfo import summary
summary(model.encoder,
        input_size=(4,5,64,2048),
        col_names=["input_size","output_size"])