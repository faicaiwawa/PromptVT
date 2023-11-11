import argparse
import torch
import _init_paths
from lib.models.PromptVT.mobileone import reparameterize_model
from lib.models.PromptVT import build_PromptVT
from lib.config.PromptVT.config import cfg, update_config_from_file
from lib.utils.box_ops import box_xyxy_to_cxcywh

import torch.nn as nn
import torch.nn.functional as F
# for onnx conversion and inference
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='PromptVT', help='script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True)

    mask = torch.rand(bs, sz, sz, requires_grad=True) > 0.5
    return img_patch, mask


class Backbone_Bottleneck_PE(nn.Module):
    def __init__(self, backbone, bottleneck_8,bottleneck_16 ,position_embed_z8,position_embed_z16):
        super(Backbone_Bottleneck_PE, self).__init__()
        self.backbone = backbone
        self.bottleneck_8 = bottleneck_8
        self.bottleneck_16 = bottleneck_16
        self.position_embed_z8 = position_embed_z8
        self.position_embed_z16 = position_embed_z16

    def forward(self, img: torch.Tensor, mask: torch.Tensor):
        feat8,feat_16 = self.backbone(img)  # BxCxHxW
        feat_8 = self.bottleneck_8(feat8)
        feat_16 = self.bottleneck_16(feat_16)
        mask_down = F.interpolate(mask[None].float(), size=feat_16.shape[-2:]).to(torch.bool)[0]

        pos_embed_z8 = self.position_embed_z8(1)  # 1 is the batch-size. output size is BxCxHxW
        pos_embed_z16 = self.position_embed_z16(1)
        # adjust shape
        feat_vec_8 = feat_8.flatten(2).permute(2, 0, 1)  # HWxBxC
        feat_vec_16 = feat_16.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_8 = pos_embed_z8.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec_16 = pos_embed_z16.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask_down.flatten(1)  # BxHW
        return feat_vec_8 , mask_vec, feat_vec_16, pos_embed_vec_8 ,pos_embed_vec_16


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "z.onnx"
    """update cfg"""
    args = parse_args()
    yaml_fname = '../experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    # build the stark model
    model = build_PromptVT(cfg, phase='test')
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = checkpoint_name = "<PATH TO PROMPTVT.PTH>"
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model.eval()
    model = reparameterize_model(model)
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck_8 = model.bottleneck_8
    bottleneck_16 = model.bottleneck_16
    position_embed_z8 = model.pos_emb_z_8
    position_embed_z16 = model.pos_emb_z_16
    torch_model = Backbone_Bottleneck_PE(backbone, bottleneck_8,bottleneck_16, position_embed_z8,position_embed_z16)
    print(torch_model)
    # get the template
    img_z, mask_z = get_data(bs, z_sz)
    # forward the template
    torch_outs = torch_model(img_z, mask_z)
    torch.onnx.export(torch_model,  # model being run
                      (img_z, mask_z),  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img_z', 'mask_z'],  # the model's input names
                      output_names=['feat_vec_8' , 'mask_vec', 'feat_vec_16', 'pos_embed_vec_8' ,'pos_embed_vec_16'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )

