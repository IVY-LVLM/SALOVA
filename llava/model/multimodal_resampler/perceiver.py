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
    def __init__(self, *, dim, dim_head=64, heads=8):
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

    def forward(self, x, latents, split_sizes=None):
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
        mask = torch.arange(max_len, device=x.device)[None, :] >= torch.tensor(split_sizes, device=x.device)[:, None]
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
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        vis_dim=256,
        cls_dim=1,
        num_latents=720,
        max_num_media=None,
        max_num_frames=2160, # None
        ff_mult=4,
    ):
        super().__init__()
        self.vis_dim = vis_dim // 4
        self.cls_dim = cls_dim
        self.num_latents = num_latents
        
        self.cls_latents = nn.Parameter(torch.randn(self.cls_dim, dim))
        self.vis_latents = nn.Parameter(torch.randn(self.num_latents, dim))
        self.latents = nn.Parameter(torch.cat([self.cls_latents, self.vis_latents], dim=0))

        self.spatial_embs = nn.Parameter(torch.randn(self.vis_dim + 1, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim))# if exists(max_num_frames) else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

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

    def forward(self, x, split_sizes=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        
        b, T, F, _ = x.shape

        # frame and media time embeddings
        if exists(self.spatial_embs):
            spatial_embs1 = repeat(self.spatial_embs[:self.vis_dim], "F d -> b T F d", b=b, T=T, F=self.vis_dim)
            x[:,:,:self.vis_dim, :] = x[:,:,:self.vis_dim, :] + spatial_embs1
            
            spatial_embs2 = repeat(self.spatial_embs[self.vis_dim:], "1 d -> b T F d", b=b, T=T, F=F-self.vis_dim)
            x[:,:,self.vis_dim:, :] = x[:,:,self.vis_dim:, :] + spatial_embs2


        if exists(self.frame_embs):
            frame_embs1 = repeat(self.frame_embs[0].unsqueeze(0), "1 d -> b T F d", b=b, T=T, F=self.vis_dim)
            x[:,:,:self.vis_dim, :] = x[:,:,:self.vis_dim, :] + frame_embs1

            frame_embs2 = repeat(self.frame_embs[1:F-self.vis_dim+1], "F d -> b T F d", b=b, T=T, F=F-self.vis_dim)
            x[:,:,self.vis_dim:F, :] = x[:,:,self.vis_dim:F, :] + frame_embs2

        # x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]
        # x.size() = torch.Size([33, 1, 92, 1024])
        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, split_sizes) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class PerceiverResampler(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.depth = model_args.mm_perceiver_depth
        self.num_latents = model_args.mm_perceiver_latents
        self.ff_mult = model_args.mm_perceiver_ff_mult
        self.vis_dim = vision_tower.num_patches - 1
        self.pretrained = model_args.mm_perceiver_pretrained

        self.perceiver = PerceiverResamplerModule(dim=vision_tower.hidden_size, depth=self.depth, num_latents=self.num_latents, vis_dim=self.vis_dim, ff_mult=self.ff_mult)

        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))

    def forward(self, image_features, split_sizes, *args, **kwargs):
            return self.perceiver(image_features, split_sizes).squeeze(1)

    @property
    def config(self):
        return {
            "mm_resampler_type": "perceiver",
            "mm_perceiver_depth": self.depth,
            "mm_perceiver_latents": self.num_latents,
            "mm_perceiver_ff_mult": self.ff_mult,
            "mm_perceiver_pretrained": self.pretrained,
        }
