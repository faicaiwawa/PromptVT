

from .backbone_X import build_backbone_x
from .position_encoding import build_position_encoding_new

from .head import build_box_head ,MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.PromptVT.exemplar_transformer import ExemplarTransformer
from lib.models.model_parts import *

from lib.models.cross_attention_encoder.hierarchical_featurefusion_network import *
from lib.models.PromptVT.DTP import DTP
import numpy as np
import time
class PromptVT(nn.Module):

    def __init__(self, backbone, transformer, box_head, pos_emb_z_16, pos_emb_z_8, pos_emb_x_16,pos_emb_x_8,head_type="CORNER_LITE",cls_head=None,distill=False,
                    e_exemplars=4,
                    sm_normalization=True,
                    temperature=2,
                    dropout=False):

        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.pos_emb_z_16 = pos_emb_z_16
        self.pos_emb_z_8 = pos_emb_z_8
        self.pos_emb_x_16 = pos_emb_x_16
        self.pos_emb_x_8 =pos_emb_x_8
        self.cls_head = cls_head
        hidden_dim = transformer.d_model
        self.bottleneck_8 = nn.Conv2d(192, hidden_dim, kernel_size=1)
        self.bottleneck_16 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.updateflag = True #when flag==true fuse the template and new dy-template
        self.fused_src_temp_8 = None
        self.fused_src_temp_16 = None
        self.head_type = head_type
        self.distill = distill
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        inchannels = 128
        outchannels_cls = 256
        padding_3 = (3 - 1) // 2

        self.branch_1 = SeparableConv2d_BNReLU(inchannels, outchannels_cls, kernel_size=3, stride=1,
                                               padding=padding_3)
        self.branch_2 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls,
                                            dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3,
                                            pw_kernel_size=1, sm_normalization=sm_normalization,
                                            temperature=temperature, dropout=dropout)
        self.branch_3 = ExemplarTransformer(in_channels=outchannels_cls, out_channels=outchannels_cls,
                                                dw_padding=padding_3, e_exemplars=e_exemplars, dw_kernel_size=3,
                                                pw_kernel_size=1, sm_normalization=sm_normalization,
                                                temperature=temperature, dropout=dropout)
        self.branch_4 = SeparableConv2d_BNReLU_out(outchannels_cls, inchannels, kernel_size=3, stride=1,
                                                   padding=padding_3)
        self.DTP_8 = DTP(inplanes=128, hide_channel=24, smooth=True)
        self.DTP_16 = DTP(inplanes=128, hide_channel=24, smooth=True)

    def PLforward(self, x):
        x = self.branch_1(x)
        x = self.branch_2(x)
        x = self.branch_3(x)
        x = self.branch_4(x)
        return x

    def forward(self, img=None, mask=None, src_temp_8=None, pos_temp_8=None, src_temp_16=None, pos_temp_16=None, src_search_8=None, pos_search_8=None, src_search_16=None ,pos_search_16=None,
                 dy_temp_8=None,dy_temp_16=None,mode="backbone", run_box_head=True, run_cls_head=True ,zx="template0", softmax=True,flag=None):
        if mode == "backbone":
            return self.forward_backbone(img, zx, mask)
        elif mode == "transformer":
            return self.forward_transformer(src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16, dy_temp_8,dy_temp_16,softmax=softmax
                                            , run_box_head=run_box_head, run_cls_head=run_cls_head,flag=flag)
        else:
            raise ValueError

    def forward_backbone(self, img: torch.Tensor, zx: str, mask: torch.Tensor):
        assert isinstance(img, torch.Tensor)
        output_back_8, output_back_16 = self.backbone(img)  # features & masks, position embedding for the search
        bs = img.size(0)  # batch size
        if zx == "search":
            pos_16 = self.pos_emb_x_16(bs)
            pos_8 = self.pos_emb_x_8(bs)
        elif "template" in zx:
            pos_8 = self.pos_emb_z_8(bs)
            pos_16 = self.pos_emb_z_16(bs)
        else:
            raise ValueError("zx should be 'template_0' or 'search'.")
        return self.adjust(output_back_8, pos_8,output_back_16,pos_16)

    def forward_transformer(self, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16, dy_temp_8,dy_temp_16,softmax=True,\
                            run_box_head=None , run_cls_head=None,flag =None,framid = None):
        '''template fusion: input[b , 2*c , h , w ] ,output [b , c , h , w ], [n ,b c ] -> [b , c ,h ,w ])'''
        self.updateflag = flag
        _framid = framid
        if (self.updateflag == True or _framid == 1):
            n , b , c = src_temp_8.shape
            self.fused_src_temp_8 = self.DTP_8(torch.cat([src_temp_8.permute(1, 2, 0).view(b, c, int(n ** 0.5) ,int(n ** 0.5)),
                                                      dy_temp_8.permute(1, 2, 0).view(b, c, int(n ** 0.5) ,int(n ** 0.5))], dim=1)).permute(2, 3, 0, 1).contiguous().view(-1, b, c)
            n, b, c = src_temp_16.shape
            self.fused_src_temp_16 = self.DTP_16(torch.cat([src_temp_16.permute(1, 2, 0).view(b, c, int(n ** 0.5) ,int(n ** 0.5)),
                                                        dy_temp_16.permute(1, 2, 0).view(b, c, int(n ** 0.5) ,int(n ** 0.5))], dim=1)).permute(2, 3, 0, 1).contiguous().view(-1, b, c)
            self.updateflag = False
            #print("dynamic template has been fused !")
        src_temp_8 = self.fused_src_temp_8
        src_temp_16 = self.fused_src_temp_16
        enc_mem = self.transformer(src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16)
        # run the corner head
        if self.distill:
            outputs_coord, prob_tl, prob_br = self.forward_box_head(enc_mem, softmax=softmax)
            return {"pred_boxes": outputs_coord, "prob_tl": prob_tl, "prob_br": prob_br}, None, None
        else:
            out, outputs_coord = self.forward_box_head(enc_mem,softmax=softmax,run_box_head=run_box_head,run_cls_head=run_cls_head)
            return out, outputs_coord, None

    def forward_box_head(self, memory, softmax=True, run_cls_head=True , run_box_head=True, ):
        """ memory: encoder embeddings (HW1+HW2, B, C) / (HW2, B, C)"""
        if "CORNER" in self.head_type:
            # encoder output for the search region (H_x*W_x, B, C)
            fx = memory[-self.feat_len_s:].permute(1,2,0).contiguous()  # (B, C, H_x*W_x)
            fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
            fx_t = self.PLforward(fx_t)  # out:torch.Size([1, 128 20, 20])
            out_dict = {}
            if run_cls_head:

                out_dict.update({'pred_logits': self.cls_head(fx_t)})
            if run_box_head:
                outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
                out_dict.update({'pred_boxes': outputs_coord})
                return out_dict ,outputs_coord
            else:
                return out_dict, None

    def adjust(self, output_back_8: torch.Tensor, pos_8: torch.Tensor, output_back_16: torch.Tensor, pos_16: torch.Tensor,):

        feat_8 = self.bottleneck_8(output_back_8)  # (B, C, H, W)
        feat_16 = self.bottleneck_16(output_back_16)  # (B, C, H, W)
        feat_vec_8 = feat_8.flatten(2).permute(2, 0, 1).contiguous()  # HWxBxC
        pos_embed_vec_8 = pos_8.flatten(2).permute(2, 0, 1).contiguous()  # HWxBxC
        feat_vec_16 = feat_16.flatten(2).permute(2, 0, 1).contiguous()  # HWxBxC
        pos_embed_vec_16 = pos_16.flatten(2).permute(2, 0, 1).contiguous()  # HWxBxC
        return {"feat_8": feat_vec_8, "pos_8": pos_embed_vec_8, "feat_16": feat_vec_16, "pos_16": pos_embed_vec_16}



def build_PromptVT(cfg, phase: str):

    backbone = build_backbone_x(cfg, phase=phase)
    transformer = build_featurefusion_network()
    box_head = build_box_head(cfg)
    fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
    pos_emb_x_16 = build_position_encoding_new(cfg, fsz_x)
    pos_emb_x_8 = build_position_encoding_new(cfg, fsz_x*2)
    pos_emb_z_16 = build_position_encoding_new(cfg, fsz_z)
    pos_emb_z_8 = build_position_encoding_new(cfg, fsz_z*2)
    cls_head = MLP(128, 256, 1, 3)
    model = PromptVT(
        backbone,
        transformer,
        box_head,
        pos_emb_z_16,
        pos_emb_z_8,
        pos_emb_x_16,
        pos_emb_x_8,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head,
        distill=cfg.TRAIN.DISTILL
    )
    return model







