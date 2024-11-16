# lsl
# 开发时间:2023/5/19 11:19
# lsl
# 开发时间:2023/5/4 16:14
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from math import exp
# 5.SSIM loss
# 生成一位高斯权重，并将其归一化
def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss/torch.sum(gauss)  # 归一化


# x=gaussian(3,1.5)
# # print(x)
# x=x.unsqueeze(1)
# print(x.shape) #torch.Size([3,1])
# print(x.t().unsqueeze(0).unsqueeze(0).shape) # torch.Size([1,1,1, 3])

# 生成滑动窗口权重，创建高斯核：通过一维高斯向量进行矩阵乘法得到
def create_window(window_size,channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size,1
    # mm:矩阵乘法 t:转置矩阵 ->1,1,window_size,_window_size
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # expand:扩大张量的尺寸，比如3,1->3,4则意味将输入张量的列复制四份，
    # 1,1,window_size,_window_size->channel,1,window_size,_window_size
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 构造损失函数用于网络训练或者普通计算SSIM值
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# 普通计算SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)