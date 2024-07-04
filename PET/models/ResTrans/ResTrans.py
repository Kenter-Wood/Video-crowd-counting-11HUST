import torch
from torch import nn
from torchvision import transforms
import math
from PIL import Image
from torch.autograd import Variable
import gc
import numpy
import torch.nn.functional as F

device = torch.device('cuda:0')

def warp(x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()  # 获取输入图像的批次大小、通道数、高度和宽度
    # 生成网格
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()  # 合并xx和yy形成二维网格
    
    if x.is_cuda:
        grid = grid.to(device)  # 如果x在GPU上，则将网格也移到GPU上
    vgrid = Variable(grid) + flo  # 根据光流更新网格位置

    # 将网格的坐标值缩放到[-1,1]范围内
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # 变换维度以适应grid_sample的输入格式
    vgrid = vgrid.to(device)
    output = nn.functional.grid_sample(x, vgrid)  # 根据新的网格采样输入图像得到变形后的图像
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = mask.to(device)
    mask = nn.functional.grid_sample(mask, vgrid)  # 同样方式采样一个全1的mask

    mask[mask < 0.9999] = 0  # 将小于0.9999的值设为0
    mask[mask > 0] = 1  # 将大于0的值设为1

    return output * mask  # 返回变形后的图像与mask的乘积

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size: int = 64, drop_p: float = 0.1, forward_expansion: int = 4,
                 forward_drop_p: float = 0.1, **kwargs):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads=4, dropout=drop_p)
        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm1(x)
        x = x + self.dropout(self.self_attention(x_norm, x_norm, x_norm)[0])
        x_norm = x + self.dropout(self.feed_forward(x_norm))
        x = x + self.dropout(self.feed_forward(x_norm))
        return x
    

class TransCE(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 8, emb_size: int = 64, depth: int = 4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        # self.position_embedding = nn.Parameter(torch.zeros(1, 1 + (img_size // patch_size) ** 2, emb_size))
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.Sequential(
            *[TransformerEncoder(emb_size) for _ in range(depth)]
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.patch_embedding(x1)
        x2 = self.patch_embedding(x2)
        x = self.transformer(torch.concat((x1, x2), dim=1))
        return x


class ResTrans(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.transce = TransCE()
        self.conv = nn.Conv2d(kernel_size=(1, 1), stride=1, padding=0, in_channels=2, out_channels=1, bias=True)

    def forward(self, Mtf: torch.Tensor, Mtv: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = Mtf.shape
        mtmean = torch.add(Mtf, Mtv) * 0.5

        output = self.transce(Mtf, Mtv)
        desired_out = (batch_size, 2, h, w)
        convin = torch.reshape(output, desired_out)
        convout = self.conv(convin)
        output = torch.add(mtmean, convout)
        return output
        return convin


class FRVCC(nn.Module):
    def __init__(self, fusion_module: nn.Module, vision_module: nn.Module, opticflow_module: nn.Module):
        super().__init__()
        self.vision = vision_module
        self.optic = opticflow_module
        self.fusion = fusion_module
        # self.first = True
        
    def get_OpticalFlow(self, It_1: torch.Tensor, It: torch.Tensor) -> torch.Tensor:
        
        to_pil = transforms.ToPILImage()
        
        batch_size = It_1.shape[0]
        h = It_1.shape[2]
        w = It_1.shape[3]
        
        of_tensor = torch.zeros((batch_size, 2, h, w))
        
        for b in range(batch_size):
            tenOne = to_pil(It_1[b])
            tenTwo = to_pil(It[b])
            
            tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(tenOne)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).to(device)
            tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(tenTwo)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).to(device)
            
            intWidth = tenOne.shape[2]
            intHeight = tenTwo.shape[1]

            tenPreprocessedOne = tenOne.view(1, 3, intHeight, intWidth)
            tenPreprocessedTwo = tenTwo.view(1, 3, intHeight, intWidth)
            
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
            
            tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

            tenFlow = torch.nn.functional.interpolate(input=self.optic(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            of_tensor[b] = tenFlow[0]
        of_tensor = of_tensor.to(device)
        return of_tensor
    
    # def get_first_Mt_1est(self, im_in: torch.Tensor) -> torch.Tensor:
    #     b, c, h, w = im_in.shape
    #     c = 1
    #     self.Mt_1est = torch.zeros((b, c, h, w)).to(device)
        
    def forward(self, It_1: torch.Tensor, It: torch.Tensor, Mtv: torch.Tensor, Mt_1est: torch.Tensor) -> torch.Tensor:
        # if self.first:
        #     self.get_first_Mt_1est(It)
        #     self.first = False
        # 计算
        Mt1_est = Mt_1est.to(device)
        OpticalFlow = self.get_OpticalFlow(It_1, It)

        # 使用原始的 OpticalFlow 进行 warp 操作
        Itest = warp(It_1, OpticalFlow).to(device)

        # 对 OpticalFlow 进行1/8降采样
        OpticalFlow_downsampled = F.interpolate(OpticalFlow, scale_factor=0.125, mode='bilinear', align_corners=False)

        # 调整为224×224大小
        OpticalFlow_resized = F.interpolate(OpticalFlow_downsampled, size=(224, 224), mode='bilinear', align_corners=False)

        # 使用降采样后的 OpticalFlow 进行 warp 操作
        Mtf = warp(Mt1_est, OpticalFlow_resized)
        
        # print(Mtf.shape)
            
        # 前向传播
        Mtest = self.fusion(Mtf, Mtv)
        self.Mt_1est = Mtest
        
        return Mtest, Itest


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TotalLoss(nn.Module):
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, Itest: torch.Tensor, It: torch.Tensor, Mtv: torch.Tensor, FRVCCout: torch.Tensor, Mtgt: torch.Tensor, D: nn.Module):
        ce = nn.MSELoss(reduction='mean')
        tv = TVLoss()
        Loss_opt = ce(Itest, It)
        # print(Loss_opt)
        Loss_opt.requires_grad_(True)
        Loss_vis = ce(Mtv, Mtgt)
        Loss_vis.requires_grad_(True)
        Loss_fus = ce(FRVCCout, Mtgt)
        # Loss_fus = ce(torch.sum(FRVCCout, dim=(1,2,3)), torch.sum(Mtgt, dim=(1,2,3)))
        Loss_fus.requires_grad_(True)
        Loss_tv = tv(FRVCCout)
        Loss_tv.requires_grad_(True)
        # Loss_G = -1 * torch.mean(torch.log(D(FRVCCout) + 1e-5 * torch.ones_like(D(FRVCCout))))
        Loss_G = -1 * torch.mean(torch.log(D(FRVCCout)))
        # print(D(FRVCCout))
        Loss_G.requires_grad_(True)
        Loss = 10 * Loss_G + 10 * Loss_fus + 5 * Loss_tv + 1 * Loss_vis + 1 * Loss_opt
        Loss.requires_grad_(True)

        return Loss_opt, Loss_vis, Loss_fus, Loss_tv, Loss_G, Loss


# desired_input = (1, 1, 224, 224)
# img = torch.randn(desired_input)
# img2 = torch.randn(desired_input)
#
# fusion_module = ResTrans()
#
# # model = FRVCC(fusion_module)
# print(fusion_module(img, img2).shape)
