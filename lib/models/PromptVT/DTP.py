
import torch.nn as nn
import torch
class Prompt(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)
        self.smooth.requires_grad=False #for test

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class DTP(nn.Module,):
    def __init__(self, inplanes=None, hide_channel=None ,smooth=True):
        super(DTP, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.prompt = Prompt(smooth=smooth)
        self.norm1 = nn.LayerNorm(inplanes)
        self.norm2 = nn.LayerNorm(inplanes)
        self.alpha = nn.Parameter(torch.zeros(1) + 0.3)
        self.alpha.requires_grad = False #for test
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        #x0 srctemp, x1 dynamictemp
        t = int(C/2) #for onnx
        x0 = x[:, 0:t, :, :].contiguous()
        x0 =x0.permute(0, 2, 3, 1).contiguous()
        _x0 = x0
        _x0 = _x0.permute(0, 3, 1, 2).contiguous()
        x0 = self.norm1(x0)
        x0 = x0.permute(0, 3, 1, 2).contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, t:, :, :].contiguous()
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.norm2(x1)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.conv0_1(x1)
        self.alpha.data.clamp_(0, 1) #for train
        x1 = self.prompt(x0) + x1
        x1 = self.conv1x1(x1)
        x0 = self.alpha * x1 + (1 - self.alpha) * _x0

        return x0



