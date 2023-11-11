import argparse
import torch
import _init_paths
from lib.utils.merge import get_qkv
from thop import profile
from thop.utils import clever_format
import time

from lib.models.PromptVT import build_PromptVT
from lib.config.PromptVT.config import cfg, update_config_from_file
from lib.models.PromptVT.mobileone import reparameterize_model

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='efttrack_s1',
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate(model, img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16):
    """Compute FLOPs, Params, and Speed"""
    # backbone
    macs1, params1 = profile(model, inputs=(img_x, att_x, None, None, None, None, "backbone", "search"),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone (search) macs is ', macs)
    print('backbone params is ', params)
    # transformer and head
    #src_temp.shape, mask_temp.shape, src_search.shape, mask_search.shape, pos_temp.shape, pos_search.shape
    #(torch.Size([64, 1, 128]), torch.Size([1, 64]), torch.Size([400, 1, 128]), torch.Size([1, 400]),
    # torch.Size([64, 1, 128]), torch.Size([400, 1, 128]))
    macs2, params2 = profile(model, inputs=(None, None, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16,
                                            src_search_8, pos_search_8, src_search_16 ,pos_search_16, dy_src_temp_8,dy_src_temp_16,"transformer",True, True,"search",True,1),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('transformer and head macs is ', macs)
    print('transformer and head params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)


def get_data(bs, sz, hw_z=64, hw_x=256, c=256):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.randn(bs, sz, sz) > 0.5
    src_temp_8 = torch.rand(256, 1, 128)
    dy_src_temp_8 = torch.rand(256, 1, 128)
    pos_temp_8 =    torch.rand(256, 1, 128)
    src_temp_16 =   torch.rand(64, 1, 128)
    dy_src_temp_16 = torch.rand(64, 1, 128)
    pos_temp_16 =   torch.rand(64, 1, 128)
    src_search_8 =  torch.rand(1600, 1, 128) #stride 8 and 16
    pos_search_8 =  torch.rand(1600, 1, 128)#stride 8 and 16
    #src_search_8 =  torch.rand(400, 1, 128)#stride 32 and 16
    #pos_search_8 =  torch.rand(400, 1, 128)#stride 32 and 16
    src_search_16 = torch.rand(400, 1, 128)
    pos_search_16 = torch.rand(400, 1, 128)
    return img_patch, att_mask, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16


if __name__ == "__main__":
    #device = "cuda:0"
    #torch.cuda.set_device(device)
    device = "cpu"
    args = parse_args()
    yaml_fname = '/home/qiuyang/PromptVT/experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE ** 2
    hw_x = cfg.DATA.SEARCH.FEAT_SIZE ** 2
    c = cfg.MODEL.HIDDEN_DIM
    model = build_PromptVT(cfg, phase='test')
    # transfer to test mode
    model.eval()
    model = reparameterize_model(model)
    print(model)
    # get the input data
    img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16 = get_data(bs, x_sz, hw_z, hw_x, c)
    # transfer to device
    model = model.to(device)
    model.box_head.coord_x = model.box_head.coord_x.to(device)
    model.box_head.coord_y = model.box_head.coord_y.to(device)
    img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16 = (img_x.to(device), att_x.to(device), \
                                              src_temp_8.to(device), pos_temp_8.to(device), src_temp_16.to(device), \
                                                pos_temp_16.to(device), src_search_8.to(device), pos_search_8.to(device), src_search_16.to(device) ,pos_search_16.to(device),\
                                                dy_src_temp_8.to(device),dy_src_temp_16.to(device))

    # evaluate the model properties
    evaluate(model, img_x, att_x,src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16)
    '''Speed Test'''
    # T_w = 50  # warmup time
    # T_t = 1000  # test time
    # print("testing speed ...")
    # with torch.no_grad():
    #     # overall
    #     for i in range(T_w):
    #         _ = model(img=img_x, mask=att_x, mode="backbone", zx="search")
    #         _ = model(src_temp_8 = src_temp_8  , pos_temp_8 = pos_temp_8, src_temp_16 = src_temp_16,
    #                                                           pos_temp_16 = pos_temp_16, src_search_8 = src_search_8, pos_search_8 = pos_search_8 ,
    #                                                           src_search_16 = src_search_16 ,pos_search_16 = pos_search_16, dy_temp_8=dy_src_temp_8,dy_temp_16=dy_src_temp_16,mode="transformer", run_box_head=True, run_cls_head=False)
    #     t_all = 0  # overall latency
    #     t_back = 0  # backbone latency
    #     for i in range(T_t):
    #         # get the input data
    #         img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_src_temp_8,dy_src_temp_16  = get_data(bs, x_sz, hw_z, hw_x, c)
    #         img_x, att_x, src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16, pos_search_16,dy_src_temp_8,dy_src_temp_16  = (
    #         img_x.to(device), att_x.to(device), \
    #         src_temp_8.to(device), pos_temp_8.to(device), src_temp_16.to(device), \
    #         pos_temp_16.to(device), src_search_8.to(device), pos_search_8.to(device), src_search_16.to(device),
    #         pos_search_16.to(device),dy_src_temp_8.to(device),dy_src_temp_16.to(device) )
    #         s = time.time()
    #         _ = model(img=img_x, mask=att_x, mode="backbone", zx="search")
    #         e_b = time.time()
    #         _ = model(src_temp_8 = src_temp_8  , pos_temp_8 = pos_temp_8, src_temp_16 = src_temp_16,
    #                                                           pos_temp_16 = pos_temp_16, src_search_8 = src_search_8, pos_search_8 = pos_search_8 ,
    #                                                           src_search_16 = src_search_16 ,pos_search_16 = pos_search_16, dy_temp_8=dy_src_temp_8,dy_temp_16=dy_src_temp_16, mode="transformer", run_box_head=True, run_cls_head=False)
    #         e = time.time()
    #         lat = e - s
    #         lat_b = e_b - s
    #         t_all += lat
    #         t_back += lat_b
    #         # print("backbone latency: %.2fms, overall latency: %.2fms" % (lat_b*1000, lat*1000))
    #     print("The average overall latency is %.2f ms" % (t_all/T_t * 1000))
    #     print("The average backbone latency is %.2f ms" % (t_back/T_t * 1000))
    #     print("The average feature fusion encoder + head latency is %.2f ms" %(t_all/T_t * 1000-t_back/T_t * 1000))


