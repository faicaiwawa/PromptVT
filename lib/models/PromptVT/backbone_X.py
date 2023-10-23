"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.PromptVT.mobileone import MobileOne
import os



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias





class Backbone(nn.Module):
    def __init__(self, name: str, dilation: bool, freeze_bn: bool, last_stage_block=14 ):
        super().__init__()

        if "RepVGG" in name:
            print("#" * 10 + "  Warning: Dilation is not valid in current code  " + "#" * 10)
            repvgg_func = get_RepVGG_func_by_name(name)
            self.body = repvgg_func(deploy=False, last_layer="stage3", freeze_bn=freeze_bn,
                                    last_stage_block=last_stage_block)
            self.num_channels = 256
        elif "convnext" in name:
            self.body = convnext_tiny(output_layers = ['layer3'], pretrained= False)
            self.num_channels = 384
        elif "mobileone" in name:
            self.body = MobileOne()
            self.num_channels = 384
        else:
            raise ValueError("Unsupported net type")

    def forward(self, x: torch.Tensor):
        return self.body(x)


def build_backbone_x_cnn(cfg, phase='train'):
    """Without positional embedding, standard tensor input"""
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN,
                        cfg.MODEL.BACKBONE.LAST_STAGE_BLOCK)
    cfg.ckpt_dir = '/home/qiuyang/efttrack_s1/lib/train/'
    if phase is 'train':
        """load pretrained backbone weights"""
        ckpt_path = None
        if hasattr(cfg, "ckpt_dir"):
            if cfg.MODEL.BACKBONE.TYPE == "RepVGG-A1":
                filename = "RepVGG-A1-train.pth"
            elif cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                filename = "LightTrackM.pth"
            elif cfg.MODEL.BACKBONE.TYPE == "convnext":
                filename = "convnext_tiny_1k_224_ema.pth"
            elif cfg.MODEL.BACKBONE.TYPE == "mobileone":
                filename = "mobileone_s1_unfused.pth.tar"
            else:
                raise ValueError("The checkpoint file for backbone type %s is not found" % cfg.MODEL.BACKBONE.TYPE)
            ckpt_path = os.path.join(cfg.ckpt_dir, filename)
        if ckpt_path is not None:
            print("Loading pretrained backbone weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                ckpt, ckpt_new = ckpt["state_dict"], {}
                for k, v in ckpt.items():
                    if k.startswith("features."):
                        k_new = k.replace("features.", "")
                        ckpt_new[k_new] = v
                ckpt = ckpt_new
            missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt, strict=False)
            #missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt['model'], strict=False)
            if is_main_process():
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)

        """freeze some layers"""
        if cfg.MODEL.BACKBONE.TYPE != "LightTrack":
            trained_layers = cfg.TRAIN.BACKBONE_TRAINED_LAYERS
            # defrost parameters of layers in trained_layers
            for name, parameter in backbone.body.named_parameters():
                parameter.requires_grad_(False)
                if train_backbone:
                    for trained_name in trained_layers:  # such as 'layer2' in layer2.conv1.weight
                        if trained_name in name:
                            parameter.requires_grad_(True)
                            break
            """
            for name, param in backbone.body.named_parameters():  #freezebn
                if 'bn' in name:
                    param.requires_grad = False
                if 'skip' in name:
                    param.requires_grad = False
            for name, param in backbone.named_parameters():
                print(name, param.requires_grad)
                """
    return backbone


def build_backbone_x_swin(cfg, phase='train'):
    """Without positional embedding, standard tensor input"""
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    if cfg.MODEL.BACKBONE.TYPE == "swin_base_patch4_window12_384_S16":
        backbone = Transformer_Backbone(cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.TYPE, train_backbone)
    else:
        raise ValueError("Unsupported model_name")

    if phase is 'train':
        """load pretrained backbone weights"""
        ckpt_path = None
        if hasattr(cfg, "ckpt_dir"):
            if cfg.MODEL.BACKBONE.TYPE == "swin_base_patch4_window12_384_S16":
                filename = "swin_base_patch4_window12_384_22k.pth"
            else:
                raise ValueError("The checkpoint file for backbone type %s is not found" % cfg.MODEL.BACKBONE.TYPE)
            ckpt_path = os.path.join(cfg.ckpt_dir, filename)
        if ckpt_path is not None:
            print("Loading pretrained backbone weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt["model"], strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del ckpt
            torch.cuda.empty_cache()
    print("For swin, we don't freeze any layers in the backbone ")
    return backbone


def build_backbone_x(cfg, phase='train'):
    if 'swin' in cfg.MODEL.BACKBONE.TYPE:
        return build_backbone_x_swin(cfg, phase=phase)
    else:
        return build_backbone_x_cnn(cfg, phase=phase)
