import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import numpy as np


def dst_process(im, num):
    transform2Tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    im_tensor = transform2Tensor(im)
    im_tensor = im_tensor * num / torch.sum(im_tensor)

    return im_tensor


def fl_process(flow_np):
    """
    将形状为 (H, W, 2) 的 NumPy 数组转换为 PIL 图像。
    
    参数：
    - flow_np: 输入的 NumPy 数组，包含光流数据，形状为 (H, W, 2)。
    
    返回：
    - 转换后的 PIL 图像。
    """
    # 提取水平和垂直分量
    flow_x = flow_np[:, :, 0]
    flow_y = flow_np[:, :, 1]
    
    # 将每个分量转换为 PIL 图像
    flow_x_img = Image.fromarray(flow_x.astype(np.uint8))
    flow_y_img = Image.fromarray(flow_y.astype(np.uint8))
    
    new_size = (flow_x_img.size[0] // 8, flow_x_img.size[1] // 8)
    flow_x_img = flow_x_img.resize(new_size, Image.ANTIALIAS)
    flow_y_img = flow_y_img.resize(new_size, Image.ANTIALIAS)

    transform2Tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    x_tensor = transform2Tensor(flow_x_img)
    y_tensor = transform2Tensor(flow_y_img)
    
    im_tensor = torch.cat((x_tensor, y_tensor), dim=0)
    return im_tensor


class MyDataset(Dataset):
    def __init__(self, img_folder, density_folder, gt_folder):
        self.img_folder = img_folder
        self.density_folder = density_folder
        # self.flow_folder = flow_folder
        self.gt_folder = gt_folder

        self.img_files = sorted(os.listdir(img_folder))
        self.density_files = sorted(glob.glob(os.path.join(density_folder, '*.jpg')))
        self.density_txt = sorted(glob.glob(os.path.join(density_folder, '*.txt')))
        # self.flow_files = sorted(os.listdir(flow_folder))
        self.gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.jpg')))
        self.gt_txt = sorted(glob.glob(os.path.join(gt_folder, '*.txt')))

        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        # self.transform_112 = transforms.Compose([
        #     transforms.Resize((112, 112)),
        #     transforms.ToTensor(),
        # ])
        self.density_process = dst_process
        self.flow_process = fl_process

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        density_file = self.density_files[idx]
        # flow_file = self.flow_folder + '/' + self.flow_files[idx]
        gt_file = self.gt_files[idx]
        density_txt_file = self.density_txt[idx]
        gt_txt_file = self.gt_txt[idx]
        img = Image.open(self.img_folder + '/' + img_file)
        density = Image.open(density_file)
        # flow = Image.open(self.flow_folder + '/' + flow_file)
        # with open(flow_file, 'rb') as f:
        #     magic = np.fromfile(f, np.float32, count=1)
        #     assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        #     h = np.fromfile(f, np.int32, count=1)[0]
        #     w = np.fromfile(f, np.int32, count=1)[0]
        #     data = np.fromfile(f, np.float32, count=2 * w * h)
        #     flow = np.resize(data, (h, w, 2))
        gt = Image.open(gt_file)

        with open(gt_txt_file, 'r') as f:
            gt_num = int(f.read().strip())
            self.gt_num = gt_num
        with open(density_txt_file, 'r') as f:
            dst_num = int(f.read().strip())
            self.pred_num = dst_num

        if idx > 0:
            prev_img_file = self.img_files[idx - 1]
            prev_density_file = self.density_files[idx - 1]
            prev_density_txt_file = self.density_txt[idx -1]
            prev_img = Image.open(self.img_folder + '/' + prev_img_file)
            prev_density = Image.open(prev_density_file)
            with open(prev_density_txt_file, 'r') as f:
                dst_num = int(f.read().strip())
                self.prev_num = dst_num
        else:
            prev_img = img.copy()
            prev_density = density.copy()
            self.prev_num = self.pred_num

        # 定义图像转换器
        transform = transforms.Compose([
            transforms.ToTensor()  # 转换为Tensor，不改变大小
        ])

        # 将图像转换为Tensor
        img_tensor = transform(img)
        prev_img_tensor = transform(prev_img)

        return img_tensor, self.density_process(density, self.pred_num), self.density_process(gt, self.gt_num), prev_img_tensor, self.density_process(prev_density, self.prev_num)


def FRVCCLoader(img_folder, density_folder, gt_folder):
    dataset = MyDataset(img_folder, density_folder, gt_folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True, num_workers=0)
    return dataloader