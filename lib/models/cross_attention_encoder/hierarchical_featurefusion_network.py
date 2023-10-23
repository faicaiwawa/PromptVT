import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
from .multi_scale_attention import *

class FeatureFusionNetwork(nn.Module):

    def __init__(self, nhead = 8, qkv_bias = True, qk_scale = None,
                 attn_drop = 0.1, drop_path = 0.1, sr_ratio = 2 ,d_model = 128):
        super().__init__()
        self.block_t_1 = Block_n1(dim = d_model, num_heads = nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, drop_path = drop_path, sr_ratio=sr_ratio)
        self.block_t_2 = Block_n2(dim = d_model, num_heads = nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, drop_path = drop_path, sr_ratio=sr_ratio )
        self.block_s_1 = Block_n1(dim = d_model, num_heads = nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, drop_path = drop_path, sr_ratio=sr_ratio)
        self.block_s_2 = Block_n2(dim = d_model, num_heads = nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, drop_path = drop_path, sr_ratio=sr_ratio)
        #self.block_final = Block_n2(dim = d_model, num_heads = nhead, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #    attn_drop=attn_drop, drop_path = drop_path, sr_ratio=sr_ratio)
        self.block_final =  finalblock()
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src_temp_stride8, pos_temp_stride8, src_temp_stride16, pos_temp_stride16,
                src_search_stride8, pos_search_stride8, src_search_stride16, pos_search_stride16):





        t_1 = self.block_t_1(self.with_pos_embed(src_temp_stride16, pos_temp_stride16),     #q-template-16 -pos
                             self.with_pos_embed(src_search_stride8, pos_search_stride8),   #k -search -8 -pos
                             self.with_pos_embed(src_search_stride16, pos_search_stride16) #k -search -16 -pos
                             )


        s_1 =self.block_s_1(self.with_pos_embed(src_search_stride16, pos_search_stride16),  #q -search -16 -pos
                            self.with_pos_embed(src_temp_stride8, pos_temp_stride8),        #k -template -8 -pos
                            self.with_pos_embed(src_temp_stride16, pos_temp_stride16)      #k -template -16 -pos
                            )

        pos_temp_stride16 = pos_temp_stride16.permute(1, 0 ,2).contiguous()
        pos_search_stride16 = pos_search_stride16.permute(1, 0 ,2).contiguous()
        t_2 = self.block_t_2(self.with_pos_embed(t_1, pos_temp_stride16),  #q -template-16- pos
                             self.with_pos_embed(s_1, pos_search_stride16),
                            20,20)

        s_2 = self.block_s_2(self.with_pos_embed(s_1, pos_search_stride16),  #q -search-16-pos
                             self.with_pos_embed(t_1, pos_temp_stride16),
                             8,8)
        pos_temp_stride16 = pos_temp_stride16.permute(1, 0, 2).contiguous()
        pos_search_stride16 = pos_search_stride16.permute(1, 0, 2).contiguous()
        t_2 = t_2.permute(1, 0, 2).contiguous()
        s_2 = s_2.permute(1, 0, 2).contiguous()
        #src_vector = self.block_final(self.with_pos_embed(s_2, pos_search_stride16),  #q -search-16-pos
        #                     self.with_pos_embed(t_2, pos_temp_stride16),8,8)
        src_vector = self.block_final(tgt=s_2, pos_dec=pos_search_stride16,
                               memory=t_2, pos_enc=pos_temp_stride16)

        return src_vector




def build_featurefusion_network():
    encoder = FeatureFusionNetwork(
        nhead = 8,
        d_model = 128
        )
    return encoder




def pltshow(pred_map,name):
    import matplotlib.pyplot as plt
    plt.figure(2)
    pred_frame = plt.gca()
    plt.imshow(pred_map,'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    pre_name = '/home/qiuyang/efttrack/heatmap/search/'+name+'.png'
    plt.savefig(pre_name,  bbox_inches='tight' , pad_inches=0 ,dpi = 300)
    plt.close(2)
