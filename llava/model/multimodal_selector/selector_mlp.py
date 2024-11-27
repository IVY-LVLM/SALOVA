import torch
import torch.nn as nn

class SelectorProjector(nn.Module):
    def __init__(self, selector_type, config):
        super().__init__()
        self._config = config

        mlp_depth = int(selector_type.group(1))
        
        modules = [nn.Linear(config.mm_selector_size, config.mm_selector_size), nn.GELU()]
        for _ in range(1, mlp_depth - 1):
            modules.append(nn.Linear(config.mm_selector_size, config.mm_selector_size))
            modules.append(nn.GELU())
        modules.append(nn.Linear(config.mm_selector_size, 1))
        
        self.proj = nn.Sequential(*modules)
        self.text_proj = nn.Linear(config.mm_text_size, config.mm_hidden_size)
        self.selector_input = config.mm_selector_input

    def cls_forward(self, image_feature, text_feature):
        text_feature = self.text_proj(text_feature)
        
        concat_embed = torch.cat([image_feature, text_feature], dim=-1)
        
        clip_size = concat_embed.size(0)
        
        return self.proj(concat_embed.view(clip_size, -1))
    
    def global_forward(self, image_feature, text_feature):
        image_feature = image_feature.mean(dim=1, keepdim=True)
        text_feature = self.text_proj(text_feature)
        
        concat_embed = torch.cat([image_feature, text_feature], dim=-1)
        
        clip_size = concat_embed.size(0)
        
        return self.proj(concat_embed.view(clip_size, -1))


    def forward(self, image_feature, text_feature, text_attn=None):
        if self.selector_input == "cls":
            output = self.cls_forward(image_feature, text_feature)
        elif self.selector_input == "global":
            output = self.global_forward(image_feature, text_feature)
        else:
            raise ValueError(f"Unexpected selector input: {self.selector_input}")
            
        return output      

        
        
    @property
    def config(self):
        return {"mm_selector_type": "selector_mlp"}
