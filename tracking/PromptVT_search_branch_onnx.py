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


def get_data(bs, sz, hw_z=64, hw_x=256, c=256):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.randn(bs, sz, sz) > 0.5
    src_temp_8 = torch.rand(256, 1, 128)
    pos_temp_8 =    torch.rand(256, 1, 128)
    src_temp_16 =   torch.rand(64, 1, 128)
    pos_temp_16 =   torch.rand(64, 1, 128)
    return img_patch,  src_temp_8, pos_temp_8, src_temp_16, pos_temp_16



class PromptVT(nn.Module):
    def __init__(self, backbone, bottleneck_8,bottleneck_16, position_embed_x8,position_embed_x16,
                 transformer, branch_1,branch_2,branch_3,branch_4,box_head,cls_head):
        super(PromptVT, self).__init__()
        self.backbone = backbone
        self.bottleneck_8 = bottleneck_8
        self.bottleneck_16 = bottleneck_16
        self.position_embed_8 = position_embed_x8
        self.position_embed_16 = position_embed_x16
        self.transformer = transformer
        self.box_head = box_head
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)
        self.branch_1 = branch_1
        self.branch_2 = branch_2
        self.branch_3 = branch_3
        self.branch_4 = branch_4
        self.box_head = box_head
        self.cls_head = cls_head



    def forward(self, img: torch.Tensor,
                feat_vec_z_8,  pos_vec_z_8,feat_vec_z_16,  pos_vec_z_16):
        # run the backbone
        feat_8,feat_16 = self.backbone(img)  # BxCxHxW
        feat_8 = self.bottleneck_8(feat_8)
        feat_16 = self.bottleneck_16(feat_16)
        #mask_down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos_embed_8 = self.position_embed_8(bs=1)  # 1 is the batch-size. output size is BxCxHxW
        pos_embed_16 = self.position_embed_16(bs=1)  # 1 is the batch-size. output size is BxCxHxW
        # adjust shape
        feat_vec_x_8 = feat_8.flatten(2).permute(2, 0, 1)  # HWxBxC
        feat_vec_x_16 = feat_16.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_vec_x_8 = pos_embed_8.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_vec_x_16 = pos_embed_16.flatten(2).permute(2, 0, 1)  # HWxBxC

        memory = self.transformer(feat_vec_z_8, pos_vec_z_8, feat_vec_z_16,pos_vec_z_16, feat_vec_x_8, pos_vec_x_8,feat_vec_x_16,pos_vec_x_16)
        fx = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
        fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
        fx_t = self.branch_1(fx_t)
        fx_t = self.branch_2(fx_t)
        fx_t = self.branch_3(fx_t)
        fx_t = self.branch_4(fx_t)
        out_dict = {}
        pred_logits = self.cls_head(fx_t)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))

        return pred_logits ,outputs_coord


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "complete.onnx"
    # update cfg
    args = parse_args()
    device = 'cpu'
    yaml_fname = '/home/qiuyang/PromptVT/experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    # build the stark model
    model = build_PromptVT(cfg, phase='test')
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = "<PATH TO PROMPTVT.PTH>"
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model.eval()
    model = reparameterize_model(model)
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck_8 = model.bottleneck_8
    bottleneck_16 = model.bottleneck_16
    position_embed_x8 = model.pos_emb_x_8
    position_embed_x16 = model.pos_emb_x_16
    transformer = model.transformer
    branch_1 = model.branch_1
    branch_2 = model.branch_2
    branch_3 = model.branch_3
    branch_4 = model.branch_4
    box_head = model.box_head
    box_head.coord_x = box_head.coord_x.cpu()
    box_head.coord_y = box_head.coord_y.cpu()
    cls_head = model.cls_head
    torch_model = PromptVT(backbone, bottleneck_8,bottleneck_16, position_embed_x8,position_embed_x16, transformer, branch_1,branch_2,branch_3,branch_4,box_head,cls_head)
    print(torch_model)
    torch.save(torch_model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    x_sz = cfg.TEST.SEARCH_SIZE
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE ** 2
    c = cfg.MODEL.HIDDEN_DIM
    print(bs, x_sz, hw_z, c)
    img_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16 = get_data(bs, x_sz)


    torch_outs = torch_model(img_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16)
    torch.onnx.export(torch_model,  # model being run
                      (img_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img_x',  'src_temp_8', 'pos_temp_8', 'src_temp_16','pos_temp_16'],  # model's input names
                      output_names=['pred_logits','outputs_coord'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # # forward the template
    # N = 1000
    # torch_model = torch_model.cuda()
    # torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    # torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()
    #
    # """########## inference with the onnx model ##########"""
    # onnx_model = onnx.load(save_name)
    # onnx.checker.check_model(onnx_model)
    # print("creating session...")
    # ort_session = onnxruntime.InferenceSession(save_name)
    # # ort_session.set_providers(["TensorrtExecutionProvider"],
    # #                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
    # print("execuation providers:")
    # print(ort_session.get_providers())
    # # compute ONNX Runtime output prediction
    # """warmup (the first one running latency is quite large for the onnx model)"""
    # for i in range(50):
    #     # pytorch inference
    #     img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda = \
    #         img_x.cuda(), mask_x.cuda(), feat_vec_z.cuda(), mask_vec_z.cuda(), pos_vec_z.cuda()
    #     torch_outs = torch_model(img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda)
    #     # onnx inference
    #     ort_inputs = {'img_x': to_numpy(img_x),
    #                   'mask_x': to_numpy(mask_x),
    #                   'feat_vec_z': to_numpy(feat_vec_z),
    #                   'mask_vec_z': to_numpy(mask_vec_z),
    #                   'pos_vec_z': to_numpy(pos_vec_z)
    #                   }
    #     s_ort = time.time()
    #     ort_outs = ort_session.run(None, ort_inputs)
    # """begin the timing"""
    # t_pyt = 0  # pytorch time
    # t_ort = 0  # onnxruntime time
    #
    # for i in range(N):
    #     # generate data
    #     img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16  = get_data(bs=bs, sz_x=sz_x, hw_z=hw_z, c=c)
    #     # pytorch inference
    #     img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16, pos_search_16, dy_src_temp_8, dy_src_temp_16 = (
    #     img_x.to(device), att_x.to(device), \
    #     src_temp_8.to(device), pos_temp_8.to(device), src_temp_16.to(device), \
    #     pos_temp_16.to(device), src_search_8.to(device), pos_search_8.to(device), src_search_16.to(device),
    #     pos_search_16.to(device), \
    #     dy_src_temp_8.to(device), dy_src_temp_16.to(device))
    #     s_pyt = time.time()
    #     torch_outs = torch_model(img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda)
    #     e_pyt = time.time()
    #     lat_pyt = e_pyt - s_pyt
    #     t_pyt += lat_pyt
    #     # print("pytorch latency: %.2fms" % (lat_pyt * 1000))
    #     # onnx inference
    #     ort_inputs = {'img_x': to_numpy(img_x),
    #                   'mask_x': to_numpy(mask_x),
    #                   'feat_vec_z': to_numpy(feat_vec_z),
    #                   'mask_vec_z': to_numpy(mask_vec_z),
    #                   'pos_vec_z': to_numpy(pos_vec_z)
    #                   }
    #     s_ort = time.time()
    #     ort_outs = ort_session.run(None, ort_inputs)
    #     e_ort = time.time()
    #     lat_ort = e_ort - s_ort
    #     t_ort += lat_ort
    #     # print("onnxruntime latency: %.2fms" % (lat_ort * 1000))
    # print("pytorch model average latency", t_pyt/N*1000)
    # print("onnx model average latency:", t_ort/N*1000)
    #
    # # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
