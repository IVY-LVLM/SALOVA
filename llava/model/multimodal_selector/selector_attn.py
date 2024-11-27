# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional
from einops import rearrange, repeat, einsum

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import torch.nn.init as init
import math
from llava.utils import rank0_print

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        # # Initialize the weights to zero
        # self.position_embeddings.weight.data.zero_()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelectorAttn(nn.Module):
    def __init__(self, config):
        # d_model=512, nhead=8, num_queries=5, num_encoder_layers=6,
        #          num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
        #          activation="relu", normalize_before=False,
        #          return_intermediate_dec=False, query_dim=2,
        #          keep_query_pos=False, query_scale_type='cond_elewise',
        #          num_patterns=0,
        #          modulate_t_attn=True,
        #          bbox_embed_diff_each_layer=False,
        #          ):
        super().__init__()
        
        self.d_model = config.selector_d_model
        self.nhead = config.selector_nhead
        self.num_encoder_layers = config.selector_num_encoder_layers
        self.dec_layers = config.selector_num_decoder_layers
        self.num_queries = config.selector_num_queries
        self.dim_feedforward = config.selector_dim_feedforward
        self.dropout = config.selector_dropout
        self.normalize_before = config.selector_normalize_before
        self.activation = config.selector_activation
        self.max_text_pe = config.max_text_pe
        self.max_video_pe = config.mm_perceiver_latents
        self.hidden_dim = config.mm_hidden_size
        self.is_double = config.selector_is_double

        t2v_encoder_layer = T2V_TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, 
                                                        self.dropout, self.activation, self.normalize_before)
        encoder_norm = nn.LayerNorm(self.d_model) if self.normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, self.num_encoder_layers, encoder_norm)

        encoder_layer = TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward,
                                                self.dropout, self.activation, self.normalize_before)
        encoder_norm = nn.LayerNorm(self.d_model) if self.normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)
    
        self.vid_pos_embed = TrainablePositionalEncoding(self.max_video_pe, self.hidden_dim)
        self.txt_pos_embed = TrainablePositionalEncoding(self.max_text_pe, self.hidden_dim)
        
        self.global_rep_token = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(self.hidden_dim))

        self.text_proj = nn.Linear(config.mm_text_size, self.hidden_dim)

        self.selector_input = config.mm_selector_input
        self.logit_scale = torch.nn.Parameter(torch.tensor([1.0]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # rank0_print(f"[*] Layer ", m)
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=0.02)  # 가중치 0.02로 초기화
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # bias 0으로 초기화
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)
    
    def forward(self, image_feature, text_feature, text_attn):
        """
        Args:
            (stage 1)
            image_feature: (batch_size, L, d) torch.Size([8, 1, 1024])
            text_feature: (batch_size, L_text, d)
            text_attn: (batch_size, L_text ) 
            gt_label
            (stage 2)
            image_feature: (batch_size, L, d) torch.Size([20, 1, 1024])
            text_feature: (batch_size, L_text, d) torch.Size([16, 146, 1024])
            text_attn: (batch_size, L_text ) 
            gt_label
        Returns:
        """
        
        # bce_loss, logits = temp_contrastive(self, image_feature.squeeze(1), text_feature[:,0,:])

        b_size = text_attn.size(0)
        src_vid = image_feature.transpose(0, 1).repeat(b_size,1,1) # [clip_num, 1, 1024] -> [bs, clip_num, 1024]
        src_txt = self.text_proj(text_feature)# [bs, text_len, 1024]
        
        src_vid_mask = torch.ones(src_vid.size(0), src_vid.size(1)).bool().to(src_vid.device) # -> [bs, clip_num]
        src_txt_mask = text_attn.bool() # -> [bs, text_len]


        src = torch.cat([src_vid, src_txt], dim=1) # [bs, clip_num + text_len, 1024]
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1)
        
        pos_vid = self.vid_pos_embed(src_vid)
        pos_txt = self.txt_pos_embed(src_txt)
        pos_embed = torch.cat([pos_vid, pos_txt], dim=1)

        # add global tokens

        mask_ = torch.tensor([[True]]).to(src_vid.device).repeat(b_size, 1) #torch.Size([32, 1])
        mask = torch.cat([mask_, mask], dim=1) #torch.Size([32, 98])
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1) 
        pos_embed_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_embed.shape[0], 1, 1)
        pos_embed = torch.cat([pos_embed_, pos_embed], dim=1)
                
        # bs, _, d = src.shape
        video_length = src_vid.size(1)
        
        src = src.transpose(0, 1)  # (L, batch_size, d)
        pos_embed = pos_embed.transpose(0, 1)   # (L, batch_size, d)

        src = self.t2v_encoder(src, src_key_padding_mask=~mask, pos=pos_embed, video_length=video_length)  # (L, batch_size, d)

        # score = temp_contrastive(self, src[1:video_length + 1].transpose(0, 1), src[video_length + 1])
        # return score

        if self.is_double:
            text_src = src[video_length + 1]
            src = src[:video_length + 1] # torch.Size([Vl, 32, 256]
            mask = mask[:, :video_length + 1]
            pos_embed = pos_embed[:video_length + 1]
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
            _, memory_local = memory[0], memory[1:]
            
            vid_src = memory_local.transpose(0, 1)  # (batch_size, L, d) condition
        else:
            vid_src = src[1:video_length + 1].transpose(0, 1)
            text_src = src[video_length + 1]

        ### v2 ###
        vid_src = vid_src / vid_src.norm(p=2, dim=-1, keepdim=True)
        text_src = text_src / text_src.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        score = einsum(vid_src, text_src, "b t d, b d -> b t") * logit_scale.to(text_src.device)
        return score
        
        # ### v3 ###
        # vid_src = vid_src / vid_src.norm(p=2, dim=-1, keepdim=True)
        # text_src = text_src / text_src.norm(p=2, dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()

        # score = (vid_src * text_src.unsqueeze(1)).sum(dim=-1) * logit_scale.to(vid_src.device)
        # ### v4 ###
        # score = self.score_proj(vid_src).squeeze()
        # return score
    
    @property
    def config(self):
        return {"mm_selector_type": "selector_attn"}

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            res = output
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            output += res
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        
        assert video_length is not None

        pos_src = self.with_pos_embed(src, pos)

        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]
        
        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)

        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):

        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        
        src2 = torch.where(torch.isnan(src2), torch.full_like(src2,0), src2)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")