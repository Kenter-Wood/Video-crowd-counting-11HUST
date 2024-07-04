import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import build_fusion, build_FRVCC, build_criterion, build_discriminator
from datasets import build_FRVCCLoader
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
        grid = grid.to(device)  # 如果x在GPU上，则将网格也移到GPU上
    vgrid = Variable(grid) + flo  # 根据光流更新网格位置

    # 将网格的坐标值缩放到[-1,1]范围内
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # 变换维度以适应grid_sample的输入格式
    output = nn.functional.grid_sample(x, vgrid)  # 根据新的网格采样输入图像得到变形后的图像
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)  # 同样方式采样一个全1的mask

    mask[mask < 0.9999] = 0  # 将小于0.9999的值设为0
    mask[mask > 0] = 1  # 将大于0的值设为1

    return output * mask  # 返回变形后的图像与mask的乘积


# 初始化模型
vision_module = build_pet()
opticflow_module = build_pwc('output/OpticalFlow.pth')
fusion_module = build_fusion()
discriminator = build_discriminator()
fusion_module.load_state_dict(torch.load('output/fusion.pth'), strict=False)
# discriminator.load_state_dict(torch.load('output/D.pth'), strict=False)
optimizer_D = Adam(discriminator.parameters(), lr=1e-6)

model = build_FRVCC(fusion_module, vision_module, opticflow_module)
# model = build_FRVCC(fusion_module, vision_module, None)

device = torch.device('cuda:0')
model.to(device)
discriminator.to(device)

optimizer_vision = Adam([{'params': [p for n, p in model.vision.named_parameters()], 'lr': 1e-4}])
optimizer_optic = Adam([{'params': [p for n, p in model.optic.named_parameters()], 'lr': 1e-4}])
optimizer_fusion = Adam([{'params': [p for n, p in model.fusion.named_parameters()], 'lr': 1e-3}])

# 初始化学习率调度器
scheduler_vision = CosineAnnealingLR(optimizer_vision, T_max=100, eta_min=0)
scheduler_optic = CosineAnnealingLR(optimizer_optic, T_max=100, eta_min=0)
scheduler_fusion = CosineAnnealingLR(optimizer_fusion, T_max=100, eta_min=0)

# 初始化损失函数
criterion = build_criterion(discriminator)

# 初始化Dataloader
img_folder = 'data/FRVCCData/images1'
# img_folder = 'WuhanMetro_train/images1'
# flow_folder = 'data/FRVCCData/opticalflow'
density_folder = 'data/FRVCCData/density_maps_eval1'
# density_folder = 'WuhanMetro_train/density_maps_eval1'
gt_folder = 'data/FRVCCData/gt1'
# gt_folder = 'WuhanMetro_train/gt1'
dataloader = build_FRVCCLoader(img_folder, density_folder, gt_folder)

is_first = True
# 训练循环
next_Mt_1est = torch.zeros((32, 1, 224, 224))
next_Mt_1est = next_Mt_1est.to(device)


# 打开一个新的文本文件来记录训练过程
with open('./training_log.txt', 'w') as f:
    for epoch in range(100):
        print(f'Epoch {epoch + 1}/100')
        for It, Mtv, Mtgt, It_1, Mt_1v in dataloader:
            # 确保所有张量在GPU上
            if is_first:
                next_Mt_1est = Mtv.to(device)
                is_first = False
            It, Mtv, Mtgt, It_1, Mt_1v = It.to(device), Mtv.to(device), Mtgt.to(device), It_1.to(device), Mt_1v.to(device)
            
            # 前向传播
            Mtest, Itest = model(It_1, It, Mtv, next_Mt_1est)
            next_Mt_1est = Mtest

            # 计算损失
            Loss_opt, Loss_vis, Loss_fus, Loss_tv, Loss_G, Loss = criterion(Itest, It, Mtv, Mtest, Mtgt, discriminator)
            
            Loss_opt = Loss_opt.mean()
            Loss_vis = Loss_vis.mean()
            Loss_fus = Loss_fus.mean()
            Loss_tv = Loss_tv.mean()
            Loss_G = Loss_G.mean()
            Loss = Loss.mean()

            # 反向传播和优化
            optimizer_fusion.zero_grad()
            optimizer_optic.zero_grad()
            Loss_fus.backward(retain_graph=True)
            Loss_opt.backward(retain_graph=True)
            Loss_G.backward(retain_graph=True)
            Loss_tv.backward()
            optimizer_fusion.step()
            optimizer_optic.step()
            
            # 更新判别器
            optimizer_D.zero_grad()
            realout = discriminator(Mtgt)
            fakeout = discriminator(Mtest.detach())
            Loss_D = 1 - realout + fakeout
            Loss_D = Loss_D.mean()
            Loss_D.backward()
            optimizer_D.step()

            # 下次计算输入
            next_Mt_1est = Mtest.detach()
            
            # 打印并写入日志
            log_msg = (f"Epoch: {epoch + 1}, Loss_opt: {Loss_opt.item()}, Loss_vis: {Loss_vis.item()}, "
                       f"Loss_fus: {Loss_fus.item()}, Loss_tv: {Loss_tv.item()}, Loss_G: {Loss_G.item()}, "
                       f"Loss: {Loss.item()}, Learning Rate: {scheduler_fusion.get_last_lr()[0]}")
            print(log_msg)
            f.write(log_msg + '\n')

        # 更新学习率
        scheduler_fusion.step()
        scheduler_vision.step()
        scheduler_optic.step()
        torch.save(model.fusion.state_dict(), './output_WM/fusion.pth')
        torch.save(model.vision.state_dict(), './output_WM/vision.pth')
        torch.save(discriminator.state_dict(), './output_WM/D.pth')
        torch.save(model.optic, './output_WM/opticflow.pth')

# 清理显存
# torch.cuda.empty_cache()
