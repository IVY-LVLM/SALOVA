#    Copyright 2023 Haotian Liu
#    Copyright 2025 Hyunjun Kim
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .multimodal_selector.builder import build_vision_selector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random, copy

import warnings
warnings.filterwarnings("ignore")

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.text_tower = build_text_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            self.mm_selector = build_vision_selector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_text_tower(self):
        text_tower = getattr(self, "text_tower", None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        text_tower = model_args.text_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.mm_text_tower = text_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")
        self.config.text_tower_pretrained = getattr(model_args, "text_tower_pretrained", "")
        self.config.train_step = getattr(model_args, "train_step", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            text_tower = build_text_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.text_tower = [text_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.text_tower = text_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
                text_tower = self.text_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
                text_tower = self.text_tower
                
            vision_tower.load_model()
            text_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_selector_type = getattr(model_args, "mm_selector_type", "linear")
        self.config.mm_selector_input = getattr(model_args, "mm_selector_input", "cls")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_selector_size = self.config.mm_hidden_size * 2
        try:
            self.config.mm_text_size = text_tower.text_tower.embeddings.position_embeddings.weight.size(1)
        except:
            self.config.mm_text_size = text_tower.text_tower.text_model.embeddings.position_embedding.weight.size(1)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        
        # selector_attn params
        self.config.selector_d_model = getattr(model_args, "selector_d_model", 256)
        self.config.selector_dropout = getattr(model_args, "selector_dropout", 0.1)
        self.config.selector_num_queries = getattr(model_args, "selector_num_queries", 5)
        self.config.selector_nhead = getattr(model_args, "selector_nhead", 8)
        self.config.selector_dim_feedforward = getattr(model_args, "selector_dim_feedforward", 1024)
        self.config.selector_num_encoder_layers = getattr(model_args, "selector_num_encoder_layers", 2)
        self.config.selector_num_decoder_layers = getattr(model_args, "selector_num_decoder_layers", 2)
        self.config.selector_normalize_before = getattr(model_args, "selector_normalize_before", True)
        self.config.selector_activation = getattr(model_args, "selector_activation", "prelu")
        self.config.max_text_pe = getattr(model_args, "max_text_pe", 2048)
        self.config.selector_is_double = getattr(model_args, "selector_is_double", True)
        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # sampler
        if getattr(self, "mm_selector", None) is None:
            self.mm_selector = build_vision_selector(self.config, vision_cfg=vision_tower.config)

            # TODO: figure out this lines
            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_selector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")
            try:
                mm_projector_weights = mm_projector_weights['module']
            except:
                pass

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
            
            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.mm_selector.load_state_dict(get_w(mm_projector_weights, "mm_selector"), strict=False)
            rank0_print(f"Loaded mm sampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.text_tower.text_tower.load_state_dict(get_w(mm_projector_weights, "text_tower.text_tower"), strict=False)
            rank0_print(f"Loaded text tower weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_clip(self, images, chunk_size=32):
        vision_tower = self.get_model().get_vision_tower()

        match (any(p.requires_grad for p in vision_tower.parameters()), torch.is_grad_enabled()):
            case (True, True):
                forward_fn = lambda x: torch.utils.checkpoint.checkpoint(vision_tower, x, use_reentrant=False)
            case _:
                forward_fn = vision_tower

        return torch.cat([forward_fn(chunk) for chunk in torch.split(images, chunk_size)])
    
    def encode_text_clip(self, text):
        text_embeds = self.get_model().get_text_tower()(text)
        return text_embeds
    
    def encode_resampler(self, image_features, resampler_idx, img_cls):
        image_features = self.get_model().vision_resampler(image_features, resampler_idx, img_cls)
        return image_features
    
    def encode_projector(self, image_features, top_idx=None, selector_input="cls"):
        local_features = self.get_model().mm_projector(image_features[:,1:,:][top_idx])
        global_features = self.get_model().mm_projector(image_features[:,0,:].unsqueeze(1))

        return local_features, global_features
    
    def encode_selector(self, image_features, text_feature, text_attn=None, selector_input="cls"):
        if selector_input == "cls":
            image_features = self.get_model().mm_selector(image_features[:,0,:].unsqueeze(1), text_feature, text_attn)
        elif selector_input == "global":
            image_features = self.get_model().mm_selector(image_features[:,1:,:], text_feature, text_attn)
        else:
            raise ValueError(f"Unexpected selector_input: {selector_input}")
        return image_features
    
    def prepare_inputs_for_resampler(self, image_features, split_sizes, clip_idx):
        image_features = torch.split(image_features, split_sizes, dim=0)
        clip_features = [torch.split(image_feature, clip, dim=0) for image_feature, clip in zip(image_features, clip_idx)]
        
        image_vis_features = [[self.get_2dPool(image_feature[0, 1:, :].unsqueeze(0)) for image_feature in clip_feature] for clip_feature in clip_features]
        image_cls_features = [[image_feature[:, 0].unsqueeze(0) for image_feature in clip_feature] for clip_feature in clip_features]

        resampler_features = [[torch.cat([vis_feature.squeeze(0), cls_feature.squeeze(0)], dim=0) for vis_feature, cls_feature in zip(image_vis_feature, image_cls_feature)] for image_vis_feature, image_cls_feature in zip(image_vis_features, image_cls_features)]
        resampler_idxes = [[r_feature.size(0) for r_feature in resampler_feature] for resampler_feature in resampler_features]
        resampler_features = [torch.nn.utils.rnn.pad_sequence(resampler_feature, batch_first=True, padding_value=0.0) for resampler_feature in resampler_features]

        return resampler_features, resampler_idxes
    
    def prepare_inputs_for_resampler_with_dropout(self, image_features, split_sizes, clip_idx):
        
        encoded_image_features = torch.split(image_features, clip_idx[0])
        image_features = []
        image_cls_features = []
        for idx, image_feat in enumerate(encoded_image_features):
            image_features.append(self.get_2dPool(image_feat[:, 1:, :]))
            if image_feat.size(0) == 1:
                image_cls_features.append(image_feat[0, 0, :].unsqueeze(0))
            else:
                image_cls_features.append(torch.mean(image_feat[:, 0, :], dim=0).unsqueeze(0))
        img_cls = torch.cat(image_cls_features, dim=0)
        
        resampler_features = [torch.nn.utils.rnn.pad_sequence(image_features, batch_first=True, padding_value=0.0)]

        return resampler_features, clip_idx, img_cls
    
    def calculate_similiarity(self, lb_sim, sen_sim, top_k=5):
        num_classes = lb_sim.size(0)
        gt_label = (sen_sim > 0.8) + (lb_sim > 0.18)
        
        if gt_label.sum() > top_k:
            topk_indices = torch.topk(lb_sim * gt_label, top_k)[1]
            gt_label = torch.zeros(num_classes, device=lb_sim.device)
            gt_label[topk_indices] = 1
        
        else:
            topk_indices = gt_label.nonzero().squeeze(1)
                
        sorted_tensor, _ = topk_indices.sort()
        
        return gt_label, sorted_tensor.tolist()
            
    # TODO: remove this
    # def calculate_similiarity(self, lb_sim, sen_sim):
    #     # Normalize lb_sim
    #     lb_sim_norm = lb_sim #(lb_sim - lb_sim.min()) / (lb_sim.max() - lb_sim.min())

    #     # Normalize sen_sim (if it's a vector, else adjust accordingly)
    #     sen_sim_norm = sen_sim #(sen_sim - sen_sim.min()) / (sen_sim.max() - sen_sim.min())

    #     # Compute entropy of lb_sim_norm
    #     # entropy_lb_sim = -torch.sum(lb_sim_norm * torch.log(lb_sim_norm + 1e-8))

    #     # Define temperature as a function of entropy (higher entropy -> higher temperature)
    #     # T = entropy_lb_sim.item() + 1e-8  # epsilon is a small constant to prevent T=0

    #     p_i = torch.softmax(lb_sim_norm / 0.5, dim=0)

    #     # Compute attention weights
    #     attention_weights = sen_sim_norm / sen_sim_norm.sum()

    #     # Adjust probabilities using attention weights
    #     p_adjusted = p_i * attention_weights

    #     # Renormalize to ensure the probabilities sum to 1
    #     p_final = p_adjusted / p_adjusted.sum()

    #     # Define threshold (e.g., keep top-k probabilities)
    #     k = min(5, p_final.size(0))  # Set based on your application
    #     topk_values, topk_indices = torch.topk(p_final, k)

    #     # Create a mask for the top-k indices
    #     mask = torch.zeros_like(p_final)
    #     mask[topk_indices] = 1

    #     # Apply the mask
    #     p_thresholded = p_final * mask

    #     # Renormalize
    #     p_final = p_thresholded / p_thresholded.sum()

    #     sorted_tensor, _ = topk_indices.sort()

    #     return p_final, sorted_tensor.tolist()
    
    # TODO: remove this
    # def encode_images(self, images):
    #     image_features = self.get_model().get_vision_tower()(images)
    #     # image_features = self.get_model().vision_resampler(image_features, images=images)
    #     image_features = self.get_model().mm_projector(image_features)
    #     return image_features
    
    # TODO: remove this (maybe not needed)
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    # TODO: remove this
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if self.config.add_faster_video:
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    # TODO: remove this
    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def temp_contrastive(self, image_feat, text_feat, target = None):
        image_embeds = self.model.tmp_visual_projection(image_feat) # [B, D]
        # image_embeds = image_feat
        text_embeds = self.model.tmp_text_projection(text_feat) # [B, D]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()

        logits = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )

        # t_loss = nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        # i_loss = nn.functional.cross_entropy(logits.t(), torch.arange(len(logits), device=logits.device))
        # return (t_loss + i_loss) / 2.0, logits
        similarity = torch.zeros(image_feat.size(0), image_feat.size(0)).to(image_feat.device)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, similarity.fill_diagonal_(1).bfloat16())

        return bce_loss, logits
    
        
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, sen_sim, lb_sim, sentences, ids, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if self.config.train_step == "sft":
            tmp_images, tmp_lb_sim, tmp_sentences, scene_nums = [], [], [], [] 
            for im, lb, sen in zip(images, lb_sim, sentences):
                scene_nums.append(len(im))
                tmp_images += im
                sen_choice = random.choice(range(len(lb)))
                tmp_lb_sim.append(lb[sen_choice])
                tmp_sentences.append(sen[sen_choice])

            images = tmp_images
            lb_sim = [tmp_lb_sim]
            sentences = [tmp_sentences]
            s_mask = torch.tensor(scene_nums) > 5

            del tmp_images
            del tmp_lb_sim
            del tmp_sentences

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                videos = []
                clip_idx = []
                for image in images:
                    if torch.is_tensor(image):
                        if image.ndim == 3:
                            image = image.unsqueeze(0)
                        videos.append(image)
                    else:
                        videos.append(torch.cat(image, dim=0))

                    clip_idx.append([instance.size(0) for instance in image])  
                        
            concat_images = torch.cat([video for video in videos], dim=0)   # [(B T), C, H, W]
            split_sizes = [image.shape[0] for image in videos]
            clip_features = self.encode_clip(concat_images)

            if self.config.train_step == 'sft' or self.config.train_step == 'pretrain':
                resampler_inputs, resampler_idxes, img_cls = self.prepare_inputs_for_resampler_with_dropout(clip_features, split_sizes, [split_sizes])
            elif self.config.train_step == 'lv':
                resampler_inputs, resampler_idxes, img_cls = self.prepare_inputs_for_resampler_with_dropout(clip_features, split_sizes, clip_idx)
            else:
                raise ValueError(f"Unexpected training step: {self.config.train_step}")

            resampler_cls = []
            local_features = []
            global_features = []
            sim_score = []
            '''
            Stage 1 short V inputs:
            resampler_inputs = List[tensor(clip_num, vid_length, hw, d)]
            clip_features.size() = torch.Size([737, 257, 1024]) (total_vid_length, hw + 1, d)
            resampler_idx = List[] sum(elements)=total_vid_length
            sentences = List[str, ... str]

            Stage 2 Long V inputs:
            resampler_inputs = List[tensor(clip_num, vid_length, hw, d)]
            clip_features.size() = torch.Size([737, 257, 1024]) (total_vid_length, hw + 1, d)
            resampler_idx = List[] sum(elements)=total_vid_length
            sentences = List[str, ... str]
            '''
            # batch inference not supported for stage 2
            assert len(resampler_inputs) == 1
            for resampler_input, resampler_idx, sentence, lb, sen in zip(resampler_inputs, resampler_idxes, sentences, lb_sim, sen_sim):
                text_embed, text_cls, text_attn = self.encode_text_clip(sentence)

                text_embed = torch.cat([text_cls.unsqueeze(1), text_embed], dim=1)
                tmp_attn = torch.ones(text_attn.size(0), 1).to(text_attn.device)
                text_attn = torch.cat([tmp_attn, text_attn], dim=1)

                resampler_feature = self.encode_resampler(resampler_input, resampler_idx, img_cls) # resampler_input.size() torch.Size([8, 35, 144, 1024])

                if self.config.train_step == "lv": # Stage 2 (Long Video) #### aldhkstjd
                    '''
                    resampler_feature.size()  torch.Size([20, 257, 1024])
                    text_embed.size()       torch.Size([16, 146, 1024])
                    gt_label.size()   torch.Size([16, 20])
                    '''     
                    gt_label = (sen > 0.83) + (lb > 0.21)
                    if gt_label[0].sum() > self.config.label_top_k:
                        topk_indices = torch.topk(lb[0], self.config.label_top_k)[1]
                        topk_mask = torch.zeros(gt_label[0].size(), device=text_attn.device)
                        topk_mask[topk_indices] = 1
                        topk_mask = topk_mask.bool()
                    else:
                        topk_mask = gt_label[0]
                    select_embed = self.encode_selector(resampler_feature, text_embed, text_attn=text_attn, selector_input=self.config.mm_selector_input)
                    gt_label = gt_label.to(select_embed.device)
                    vis_embed, global_embed = self.encode_projector(resampler_feature, top_idx=topk_mask, selector_input=self.config.mm_selector_input)
                    gt_label = gt_label.float()
                    resampler_cls.append(select_embed)
                    sim_score.append(gt_label)

                elif  self.config.train_step == "sft":
                    rf = torch.split(resampler_feature, scene_nums)
                    vis_embed, global_embed = [], []

                    for rfi, tei, tai, lbi, sci in zip(rf, text_embed, text_attn, lb, scene_nums):
                        gt_label = (torch.tensor(lbi) > 0.20).float()
                        if gt_label.sum() == 0:
                            gt_label[lbi.index(max(lbi))] = 1.0
                        select_embed = self.encode_selector(rfi, tei.unsqueeze(0), text_attn=tai.unsqueeze(0), selector_input=self.config.mm_selector_input)
                        resampler_cls.append(select_embed)
                        sim_score.append(gt_label.unsqueeze(0).to(select_embed.device))
                        top_idx = None
                        if sci > self.config.label_top_k:
                            _, topk_indices = torch.topk(torch.tensor(lbi), self.config.label_top_k)
                            top_idx = torch.zeros(torch.tensor(lbi).size(), dtype=torch.bool)
                            top_idx[topk_indices] = True

                        L, G = self.encode_projector(rfi, top_idx=top_idx, selector_input=self.config.mm_selector_input) 
                        L = L.squeeze(0)
                        
                        vis_embed.append(L)
                        global_embed.append(G)

                elif  self.config.train_step == "pretrain":
                    '''
                    resampler_feature.size()  torch.Size([8, 257, 1024])
                    text_embed.size()       torch.Size([8, 22, 1024])
                    '''
                    select_embed = self.encode_selector(resampler_feature, text_embed, text_attn=text_attn, selector_input=self.config.mm_selector_input)
                    resampler_cls.append(select_embed)

                    # resampler_feature.size() torch.Size([8, 65, 1024])
                    similarity = torch.zeros(select_embed.size(0), select_embed.size(0)).to(select_embed.device)
                    sim_score.append(similarity.fill_diagonal_(1))
                    
                    vis_embed, global_embed = self.encode_projector(resampler_feature, top_idx=None, selector_input=self.config.mm_selector_input) 
                    vis_embed = vis_embed.squeeze(0)
                
                local_features.append(vis_embed)
                                
                if global_embed is not None:
                    global_features.append(global_embed)
            
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            # image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            # mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                if global_features is not None:
                    if self.config.train_step == 'pretrain':
                        image_features = [torch.cat([x, y], dim = 0) for x,y in zip(local_features[0], global_features[0])]
                    elif self.config.train_step == 'lv':
                        image_features = [torch.cat([x.flatten(0, 1), y.flatten(0,1)],) for x,y in zip(local_features, global_features)]
                    elif self.config.train_step == 'sft':
                        image_features = [torch.cat([x.flatten(0, 1), y.flatten(0, 1)], dim = 0) for x,y in zip(local_features[0], global_features[0])]
                    else:
                        raise ValueError(f"Unexpected training step: {self.config.train_step}")
                else:
                    image_features = [x.flatten(0, 1) for x in local_features]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, resampler_cls, sim_score, ids

    def prepare_inputs_labels_for_multimodal_infer(self, input_ids, position_ids, attention_mask, past_key_values, labels, sentences, images, modalities=["image"], image_sizes=None, longlong=False):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if longlong:
            resampler_container = []
            for image in images:
                seg_images = [image]
                if type(seg_images) is list:
                    videos = []
                    clip_idx = []
                    for image in seg_images:
                        if torch.is_tensor(image):
                            if image.ndim == 3:
                                image = image.unsqueeze(0)
                            videos.append(image)
                        else:
                            videos.append(torch.cat(image, dim=0))

                        clip_idx.append([instance.size(0) for instance in image])  
                            
                concat_images = torch.cat([video for video in videos], dim=0)   # [(B T), C, H, W] torch.Size([9, 3, 336, 336])
                split_sizes = [image.shape[0] for image in videos] # [9]
                clip_features = self.encode_clip(concat_images) # torch.Size([9, 577, 1024])

                # TODO: wrote by JH: only applied on batch size 1
                resampler_inputs, resampler_idxes, img_cls = self.prepare_inputs_for_resampler_with_dropout(clip_features, split_sizes, [split_sizes])
                # resampler_inputs[0].size() = torch.Size([1, 9, 144, 1024])
                # resampler_idxes = [[9]]
                # img_cls.size() = torch.Size([1, 1024])

                # batch inference not supported for stage 2
                assert len(resampler_inputs) == 1
                resampler_input = resampler_inputs[0]
                resampler_idx = resampler_idxes[0]

                resampler_feature = self.encode_resampler(resampler_input, resampler_idx, img_cls) # resampler_input.size() torch.Size([8, 35, 144, 1024])
                resampler_container.append(resampler_feature)
            resampler_feature = torch.cat(resampler_container, dim = 0)
        else:
            if type(images) == dict:
                def divide_number(n, m):
                    quotient = n // m 
                    remainder = n % m 

                    result = [quotient + 1 if i < remainder else quotient for i in range(m)]
                    return result
                
                def fix_num(split_sizes, cut_index):

                    total_splits = len(split_sizes)
                    total_frames = sum(split_sizes)
                    R = cut_index/total_frames
                    first = round(total_splits * R)
                    second = total_splits - first
                    if cut_index == 0:
                        return [1] + divide_number(total_frames-1, total_splits)
                    if cut_index+1 == total_frames:
                        return divide_number(total_frames-1, total_splits) + [1]
                    if first == 0:
                        first = 1
                        second = total_splits - 1
                    elif second == 0:
                        second = 1
                        first = total_splits - 1
                    cut_split_sizes = divide_number(cut_index, first) + [1] + divide_number(total_frames-cut_index-1, second)
                    return cut_split_sizes
                
                clip_features=images['image']

                if clip_features.size(0) > 1000:
                    split_sizes=divide_number(clip_features.size(0), 10)
                elif clip_features.size(0) > 500:
                    split_sizes=divide_number(clip_features.size(0), 7)
                elif clip_features.size(0) > 200:
                    split_sizes=divide_number(clip_features.size(0), 4)
                else:
                    split_sizes=divide_number(clip_features.size(0), 3)
                # else:
                #     split_sizes=divide_number(clip_features.size(0), clip_features.size(0))
                # cut_index=images['segment']
                # split_sizes = fix_num(split_sizes, cut_index)

            else:
                videos = []
                clip_idx = []
                for image in images:
                    if torch.is_tensor(image):
                        if image.ndim == 3:
                            image = image.unsqueeze(0)
                        videos.append(image)
                    else:
                        videos.append(torch.cat(image, dim=0))

                    clip_idx.append([instance.size(0) for instance in image])  
                        
                concat_images = torch.cat([video for video in videos], dim=0)   # [(B T), C, H, W] torch.Size([9, 3, 336, 336])
                split_sizes = [image.shape[0] for image in videos] # [9]
                clip_features = self.encode_clip(concat_images) # torch.Size([9, 577, 1024])


            # TODO: wrote by JH: only applied on batch size 1
            resampler_inputs, resampler_idxes, img_cls = self.prepare_inputs_for_resampler_with_dropout(clip_features, split_sizes, [split_sizes])
            # resampler_inputs[0].size() = torch.Size([1, 9, 144, 1024])
            # resampler_idxes = [[9]]
            # img_cls.size() = torch.Size([1, 1024])

            # batch inference not supported for stage 2
            assert len(resampler_inputs) == 1
            resampler_input = resampler_inputs[0]
            resampler_idx = resampler_idxes[0]

            resampler_feature = self.encode_resampler(resampler_input, resampler_idx, img_cls) # resampler_input.size() = torch.Size([3, 22, 144, 1024])
            # resampler_feature.size() = torch.Size([1, 257, 1024])
        
        local_features = []
        global_features = []
        sentence = sentences[0]
        text_embed, text_cls, text_attn = self.encode_text_clip(sentence)
        text_embed = torch.cat([text_cls.unsqueeze(1), text_embed], dim=1)
        tmp_attn = torch.ones(text_attn.size(0), 1).to(text_attn.device)
        text_attn = torch.cat([tmp_attn, text_attn], dim=1)

        select_embed = self.encode_selector(resampler_feature, text_embed, text_attn=text_attn, selector_input=self.config.mm_selector_input)
        
        # TODO: wrote by JH: select top k indices and put them into encoder_projector arguments
        top_k_num = min(select_embed.size(-1), self.config.label_top_k)
        topk_indices = torch.topk(select_embed[0], top_k_num)[1]
        topk_indices = topk_indices.sort()[0]

        vis_embed, global_embed = self.encode_projector(resampler_feature, top_idx=topk_indices, selector_input=self.config.mm_selector_input)

        local_features.append(vis_embed)
                        
        if global_embed is not None:
            global_features.append(global_embed)
        
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")

        if mm_patch_merge_type == "flat":
            if global_features is not None:
                image_features = [torch.cat([x.flatten(0, 1), y.flatten(0,1)],) for x,y in zip(local_features, global_features)]
            else:
                image_features = [x.flatten(0, 1) for x in local_features]
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        '''
        image_features = List(torch.tensor(L, D), torch.tensor(L, D), ...)
        '''
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")
        
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
