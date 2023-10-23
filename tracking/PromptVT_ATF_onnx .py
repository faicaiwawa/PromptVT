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
    src_temp_8 = torch.rand(256, 1, 128)
    src_temp_16 =   torch.rand(64, 1, 128)
    dy_src_temp_8 = torch.rand(256, 1, 128)
    dy_src_temp_16 =   torch.rand(64, 1, 128)
    return src_temp_8, src_temp_16, dy_src_temp_8, dy_src_temp_16


class AFF(nn.Module):
    def __init__(self, AFF_8,AFF_16):
        super(AFF, self).__init__()
        self.AFF_8 = AFF_8
        self.AFF_16 = AFF_16

    def forward(self, src_temp_8, src_temp_16, dy_src_temp_8, dy_src_temp_16):

        n, b, c = src_temp_8.shape
        t = int(n ** 0.5)
        con_temp_8 = torch.cat([src_temp_8.permute(1, 2, 0).view(b, c, t, t),
                   dy_src_temp_8.permute(1, 2, 0).view(b, c, t, t)], dim=1)

        n, b, c = src_temp_16.shape
        t = int(n ** 0.5)
        con_temp_16 = torch.cat([src_temp_16.permute(1, 2, 0).view(b, c, t, t),
                   dy_src_temp_16.permute(1, 2, 0).view(b, c, t, t)], dim=1)

        fused_temp_8 = AFF_8(con_temp_8).permute(2, 3, 0, 1).contiguous().view(-1, b, c)

        fused_temp_16 = AFF_16(con_temp_16).permute(2, 3, 0, 1).contiguous().view(-1, b, c)

        return  fused_temp_8, fused_temp_16


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "AFF.onnx"
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
        checkpoint_name = checkpoint_name = "/home/qiuyang/PromptVT/checkpoints/PromptVT/baseline/standar_vipt_channel_24/b_updated.pth"
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model.eval()
    model = reparameterize_model(model)
    """ rebuild the inference-time model """
    AFF_8 = model.AdaptiveFusion_8
    AFF_16 = model.AdaptiveFusion_16
    torch_model = AFF(AFF_8,AFF_16)
    print(torch_model)
    # get the template
    src_temp_8, src_temp_16, dy_src_temp_8, dy_src_temp_16 = get_data(bs, z_sz)


    # forward the template
    fused_temp_8, fused_temp_16 = torch_model(src_temp_8, src_temp_16, dy_src_temp_8, dy_src_temp_16)
    torch.onnx.export(torch_model,  # model being run
                      (src_temp_8, src_temp_16, dy_src_temp_8, dy_src_temp_16),  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['src_temp_8', 'src_temp_16', 'dy_src_temp_8', 'dy_src_temp_16'],  # the model's input names
                      output_names=['fused_temp_8' ,  'fused_temp_16'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    # latency comparison
    # N = 1000
    # """########## inference with the pytorch model ##########"""
    # torch_model = torch_model.cuda()
    # s = time.time()
    # for i in range(N):
    #     img_z_cuda, mask_z_cuda = img_z.cuda(), mask_z.cuda()
    #     _ = torch_model(img_z_cuda, mask_z_cuda)
    # e = time.time()
    # print("pytorch model average latency: %.2f ms" % ((e - s) / N * 1000))
    # """########## inference with the onnx model ##########"""
    # onnx_model = onnx.load(save_name)
    # onnx.checker.check_model(onnx_model)
    #
    # ort_session = onnxruntime.InferenceSession(save_name)
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {'img_z': to_numpy(img_z),
    #               'mask_z': to_numpy(mask_z)}
    # # print(onnxruntime.get_device())
    # # warmup
    # for i in range(10):
    #     ort_outs = ort_session.run(None, ort_inputs)
    # s = time.time()
    # for i in range(N):
    #     ort_outs = ort_session.run(None, ort_inputs)
    # e = time.time()
    # print("onnx model average latency: %.2f ms" % ((e - s) / N * 1000))
    # # compare ONNX Runtime and PyTorch results
    # for i in range(3):
    #     np.testing.assert_allclose(to_numpy(torch_outs[i]), ort_outs[i], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
