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
from lib.models.PromptVT.adaptivefusionmodule import  AdaptiveFusion




def evaluate(model, f):
    """Compute FLOPs, Params, and Speed"""
    # backbone
    macs1, params1 = profile(model, inputs=(f), custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print(' macs is ', macs)
    print(' params is ', params)



def get_data(bs, sz, hw_z=64, hw_x=256, c=256):
    F = torch.rand(1,1,512, 8, 8)
    return F


if __name__ == "__main__":
    #device = "cuda:0"
    #torch.cuda.set_device(device)
    device = "cpu"
    # Compute the Flops and Params of our STARK-S model


    '''set some values'''
    bs = 1

    # build the stark model
    model  = AdaptiveFusion(inplanes=256, hide_channel=24, smooth=True)

    model.eval()

    print(model)

    f= get_data(bs,2)

    model = model.to(device)

    f = f.to(device)


    evaluate(model, f)


