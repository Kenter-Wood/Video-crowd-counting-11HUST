import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import build_fusion, build_FRVCC, build_criterion, build_discriminator
from torchvision import transforms
from datasets import build_FRVCCLoader
import numpy as np
from PET_module import build_pet
from torch.autograd import Variable
import torch.nn as nn
from pytorch_pwc_master import build_pwc, Network, Extractor, Decoder, Refiner


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
        grid = grid.cuda()  # 如果x在GPU上，则将网格也移到GPU上
    vgrid = Variable(grid) + flo  # 根据光流更新网格位置

    # 将网格的坐标值缩放到[-1,1]范围内
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # 变换维度以适应grid_sample的输入格式
    output = nn.functional.grid_sample(x, vgrid)  # 根据新的网格采样输入图像得到变形后的图像
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)  # 同样方式采样一个全1的mask

    mask[mask < 0.9999] = 0  # 将小于0.9999的值设为0
    mask[mask > 0] = 1  # 将大于0的值设为1

    return output * mask  # 返回变形后的图像与mask的乘积


# 初始化模型
vision_module = build_pet()
opticflow_module = build_pwc('output_WM/opticflow.pth')
fusion_module = build_fusion()
discriminator = build_discriminator()
vision_module.load_state_dict(torch.load('output_WM/vision.pth'), strict=False)
fusion_module.load_state_dict(torch.load('output_WM/fusion.pth'), strict=False)
discriminator.load_state_dict(torch.load('output_WM/D.pth'))

# model = build_FRVCC(fusion_module, vision_module, opticflow_module)
model = build_FRVCC(fusion_module, vision_module, opticflow_module)

device = torch.device('cuda:0')
model.to(device)


# 初始化Dataloader
img_folder = 'data/FRVCCData/images1'
# flow_folder = 'WuhanMetro_train/output1'
density_folder = 'data/FRVCCData/density_maps_eval1'
gt_folder = 'data/FRVCCData/gt1'
dataloader = build_FRVCCLoader(img_folder, density_folder, gt_folder)

toPIL = transforms.ToPILImage()
next_Mt_1est = torch.zeros((4, 1, 224, 224)).to(device)

is_first = True

with torch.no_grad():
    for It, Mtv, Mtgt, It_1, Mt_1v in dataloader:
        # if is_first:
        #         next_Mt_1est = Mtv.to(device)
        #         is_first = False
        # 确保所有张量在GPU上
        It, Mtv, Mtgt, It_1, Mt_1v = It.to(device), Mtv.to(device), Mtgt.to(device), It_1.to(device), Mt_1v.to(device)
            
        # 前向传播
        Mtest, Itest = model(It_1, It, Mtv, next_Mt_1est)
        
        b, c, h, w = Mtv.shape
        resize = transforms.Resize((h, w))
        to_pil = transforms.ToPILImage()
        for i in range(It.size(0)):
            img_tensor = Mtest[i]  # 获取单个图像 (C, H, W)

            # 调整大小
            img_resized = resize(img_tensor)

            # 转换为 PIL 图像
            img_pil = to_pil(img_resized)

            # 保存图像
            img_pil.save(f'epoch/epoch1/image_{i}.jpg')

        print("图像已保存")
        next_Mt_1est = Mtest
        # Mtest.to(device)

        # 计算损失
        # Loss_opt, Loss_vis, Loss_fus, Loss_tv, Loss = criterion(Itest, It, Mtv, Mtest, Mtgt)
        # Mtest = model(Mtf, Mtv)
        # print('Mtest',Mtest)
        # 下次计算输入
        next_Mt_1est = Mtest.to(device)
        num_gt = torch.sum(Mtgt, dim=(1, 2, 3))
        num_est = torch.sum(Mtest, dim=(1, 2, 3))
        num_vis = torch.sum(Mtv, dim=(1, 2, 3))
        # print('gt:', num_gt)
        # print('pred:', num_est)
        # print('vision_pred', num_vis)
        print('FRVCCMAE', torch.mean(torch.abs(num_gt - num_est)))
        print('PETMAE', torch.mean(torch.abs(num_gt - num_vis)))
        # print('of', torch.sum(OpticFlow, dim=(1,2,3)))