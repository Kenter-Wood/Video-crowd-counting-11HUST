from .backbones import *
from .transformer import *
from .ResTrans import *
from .pet import build_pet
import torch.nn as nn


def build_model(args):
    return build_pet(args)


def build_fusion():
    return ResTrans()


def build_FRVCC(fusion, vision=None, optic=None):
    return FRVCC(fusion, vision, optic)


def build_criterion(discriminator: nn.Module):
    return TotalLoss(discriminator)


def build_discriminator():
    model = vgg16_bn(pretrained=True)
    # 修改输入通道
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # 获取模型的输出维度
    num_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 1),  # 输出一个值
    nn.Sigmoid()  # 添加 sigmoid 激活函数
    )

    return model
