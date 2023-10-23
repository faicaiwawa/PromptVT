import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.PromptVT.backbone_X import FrozenBatchNorm2d




def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class Corner_Predictor_Lite_Rep_v2(nn.Module):


    def __init__(self, inplanes=128, channel=128, feat_sz=20, stride=16):
        super(Corner_Predictor_Lite_Rep_v2, self).__init__()
        self.feat_sz = feat_sz
        self.feat_len = feat_sz ** 2
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''convolution tower for two corners'''
        self.conv_tower = nn.Sequential(nn.Conv2d(128, 2, kernel_size=3, padding=1))


        '''for gpu'''
        #with torch.no_grad():
        #    self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
        #    # generate mesh-grid
        #    self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
        #        .view((self.feat_sz * self.feat_sz,)).float().cuda()
        #    self.coord_y = self.indice.repeat((1, self.feat_sz)) \
        #        .view((self.feat_sz * self.feat_sz,)).float().cuda()

        with torch.no_grad():
            self.indice = (torch.arange(0, self.feat_sz).view(-1, 1) + 0.5) * self.stride  # here we can add a 0.5
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float()

    def forward(self, x):

        score_map_tl, score_map_br = self.get_score_map(x)

        # if return_dist:
        #     coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
        #     coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
        #     return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br

        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)

        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map = self.conv_tower(x)  # (B,2,H,W)
        return score_map[:, 0, :, :].contiguous(), score_map[:, 1, :, :]

    def get_score_map_v2(self, x):
        score_map1 = self.conv_tower_v2(x)  # (B,1,H,W) top-left corner heatmap
        score_map2 = self.conv_tower_v2(x)  # (B,1,H,W) bottom-right corner heatmap
        return score_map1, score_map2


    #def soft_argmax(self, score_map, return_dist=False, softmax=True):
    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_len))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        #if return_dist:
        #    if softmax:
        #        return exp_x, exp_y, prob_vec
        #    else:
        #        return exp_x, exp_y, score_vec

        return exp_x, exp_y



def build_box_head(cfg):
    if "CORNER" in cfg.MODEL.HEAD_TYPE:
        if cfg.MODEL.BACKBONE.DILATION is False:
            stride = 16
        else:
            stride = 8
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        #print("head channel: %d" % channel)

        if cfg.MODEL.HEAD_TYPE == "CORNER_LITE_REP_v2":
            corner_head = Corner_Predictor_Lite_Rep_v2(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                                       feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        #self.conv = nn.Conv2d(128, 256, kernel_size=20, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        b , c , h ,w = x.shape
        #x=x.view(b,c,int(n**0.5),int(n**0.5))
        x=self.avg_pooling(x).view(b,c)

        #x=x.flatten(1);
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x #[3,1]


