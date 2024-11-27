"""
Taken from https://github.com/lucidrains/flamingo-pytorch
"""

import torch
from einops import rearrange, repeat

try:
    from einops_exts import rearrange_many
except:
    pass

from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=2):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """

        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # Create and apply the mask
        max_len = x.size(-2) # x.shape[-2]
        mask_len = v.size(-2)
        mask = torch.cat([mask, torch.zeros(mask_len - max_len, device=x.device, dtype=torch.bool).repeat(x.size(0), 1)], dim=-1)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, h, -1, latents.shape[-2], -1) # mask[:, None, None, None].expand(-1, h, -1, latents.size(-2), -1)
        
        sim.masked_fill_(mask, float('-inf'))  # Apply the mask to the similarity scores

        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResamplerModule(nn.Module):
    def __init__(
        self,
        *,
        dim=1024,
        depth=2,
        dim_head=64,
        heads=2,
        vis_dim=256,
        cls_dim=1,
        num_latents=720,
        max_num_media=None,
        max_num_frames=2160, # None
        patch_drop_rate=None, # None
        ff_mult=4,
    ):
        super().__init__()
        self.vis_dim = vis_dim // 4
        if dim == 1152:
            self.vis_dim = 196
        self.cls_dim = cls_dim
        self.num_latents = num_latents
        self.patch_drop_rate = patch_drop_rate
        self.cls_latents = nn.Parameter(torch.zeros(self.cls_dim, dim))
        self.vis_latents = nn.Parameter(torch.randn(self.num_latents, dim))
        self.latents = nn.Parameter(torch.cat([self.cls_latents, self.vis_latents], dim=0))

        self.spatial_embs = nn.Parameter(torch.randn(self.vis_dim, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim))# if exists(max_num_frames) else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None
        self.max_num_frames = max_num_frames

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult) if ff_mult > 0 else nn.Identity(),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, split_sizes=None, img_cls = None):
        """
        Args:
            x (torch.Tensor): image features
                shape (T, F, hw, D)

        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        
        T, F, v, d = x.shape
        split_sizes = [x*self.vis_dim for x in split_sizes]

        # frame and media time embeddings
        if exists(self.spatial_embs):
            x = x + repeat(self.spatial_embs, "v d -> T F v d", T=T, F=F)

        if exists(self.frame_embs):
            if F > self.max_num_frames:
                drop_indices = torch.randperm(F)[:self.max_num_frames]
                x = x[:, drop_indices, :, :]
                
            x = x + repeat(self.frame_embs[:F], "F d -> T F v d", T=T, v=v)

        x = rearrange(x, "T F v d -> T 1 (F v) d")  # flatten the frame and spatial dimensions

        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # Generate mask
        seq_len = x.size(2)
        out_mask = torch.arange(seq_len, device=x.device)[None, :] >= torch.tensor(split_sizes, device=x.device)[:, None]

        if self.patch_drop_rate:
            tmp_mask = []
            tmp_list = []
            if seq_len > 18000:
                dynamic_ratio = 1.0
            else:
                dynamic_ratio = (seq_len/18000)
            
            num_keep = max(1, int(seq_len * (1 - (dynamic_ratio*self.patch_drop_rate))))

            for n in range(T):
                if split_sizes[n] <= num_keep:
                    idx = torch.arange(num_keep)
                else:
                    idx = torch.randperm(split_sizes[n])[:num_keep]

                mask = torch.zeros(seq_len, dtype=torch.bool)
                mask[idx] = True

                if len(mask) != x[n].size(1):
                    print("*** perceiverv2 dimension error")
                    print(f"mask: {mask.size()} x[{n}]: {x[n].size()}")
                tmp_list.append(x[n][:, mask, :].unsqueeze(0))
                tmp_mask.append(out_mask[n][mask].unsqueeze(0))
            x = torch.cat(tmp_list, dim=0)
            out_mask = torch.cat(tmp_mask, dim=0)

        # blocks
        if img_cls is not None:
            b_latent = repeat(self.latents, "n d -> T n d", T=T)
            cls_latent = b_latent[:, 0, :] + img_cls
            latents = torch.cat([cls_latent.unsqueeze(1), b_latent[:, 1:, :]], dim = 1).unsqueeze(1)
        else:
            latents = repeat([cls_latent, self.latents], "n d -> T 1 n d", T=T)
        
        for attn, ff in self.layers:
            latents = attn(x, latents, out_mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class PerceiverResamplerv2(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.heads = model_args.mm_perceiver_heads
        self.depth = model_args.mm_perceiver_depth
        self.num_latents = model_args.mm_perceiver_latents
        self.ff_mult = model_args.mm_perceiver_ff_mult
        self.vis_dim = vision_tower.num_patches - 1
        self.pretrained = model_args.mm_perceiver_pretrained
        self.patch_dropout_rate = model_args.patch_dropout_rate
        self.frame_dropout = model_args.frame_dropout

        self.perceiver = PerceiverResamplerModule(dim=vision_tower.hidden_size, 
                                                  heads = self.heads, 
                                                  depth=self.depth, 
                                                  num_latents=self.num_latents, 
                                                  vis_dim=self.vis_dim, 
                                                  ff_mult=self.ff_mult, 
                                                  patch_drop_rate=self.patch_dropout_rate)

        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))

    def forward(self, image_features, split_sizes, *args, **kwargs):
            return self.perceiver(image_features, split_sizes, *args, **kwargs).squeeze(1)

    @property
    def config(self):
        return {
            "mm_resampler_type": "perceiverv2",
            "mm_perceiver_depth": self.depth,
            "mm_perceiver_latents": self.num_latents,
            "mm_perceiver_ff_mult": self.ff_mult,
            "mm_perceiver_heads": self.heads,
            "mm_perceiver_pretrained": self.pretrained,
        }