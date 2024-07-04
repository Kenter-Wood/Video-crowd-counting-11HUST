import h5py
import json
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet2s
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms

from sklearn.metrics import mean_squared_error,mean_absolute_error

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the json file contains path of test images
test_json_path = './test.json'

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)



model = CANNet2s()

model = model.cuda()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load('checkpoint.pth.tar')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred= []
gt = []

with open('results.txt', 'w') as file:  # 'a' 模式表示追加写入
    for i in range(len(img_paths)):
        targets = 0
        img_path = img_paths[i]

        img_folder = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        print(img_folder + '/' + img_name)
        index = int(img_name.split('.')[0])

        prev_index = int(max(1,index-5))

        prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))

        prev_img = Image.open(prev_img_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        prev_img = prev_img.resize((640,360))
        img = img.resize((640,360))

        prev_img = transform(prev_img).cuda()
        img = transform(img).cuda()

        gt_path = img_path.replace('.jpg','_resize.h5')
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)


        img = img.unsqueeze(0)
        prev_img = prev_img.unsqueeze(0)

        prev_flow = model(prev_img,img)

        prev_flow_inverse = model(img,prev_img)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0,:] = 1.0
        mask_boundry[-1,:] = 1.0
        mask_boundry[:,0] = 1.0
        mask_boundry[:,-1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0,0,1:,1:],(0,1,0,1))+F.pad(prev_flow[0,1,1:,:],(0,0,0,1))+F.pad(prev_flow[0,2,1:,:-1],(1,0,0,1))+F.pad(prev_flow[0,3,:,1:],(0,1,0,0))+prev_flow[0,4,:,:]+F.pad(prev_flow[0,5,:,:-1],(1,0,0,0))+F.pad(prev_flow[0,6,:-1,1:],(0,1,1,0))+F.pad(prev_flow[0,7,:-1,:],(0,0,1,0))+F.pad(prev_flow[0,8,:-1,:-1],(1,0,1,0))+prev_flow[0,9,:,:]*mask_boundry


        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0,:9,:,:],dim=0)+prev_flow_inverse[0,9,:,:]*mask_boundry


        overall = ((reconstruction_from_prev+reconstruction_from_prev_inverse)/2.0).data.cpu().numpy()
        target = target
        # 可视化
        gray_image = np.array(overall)

        # 进行灰度映射，将最小值映射为0，最大值映射为255
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)

        # 避免除零错误
        if max_val - min_val != 0:
            scaled_image = 255 * (gray_image - min_val) / (max_val - min_val)
        else:
            scaled_image = gray_image

        # 将数据类型转换为无符号8位整型
        scaled_image = scaled_image.astype(np.uint8)

        # cv2.imwrite(f'vis/{img_name}', scaled_image)
        # 可视化
        plt.imshow(overall, cmap='jet')  # 使用 'jet' 颜色映射
        plt.axis('off')

        # 创建保存图像的文件夹
        output_folder = 'vis_overall'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 根据索引 i 生成文件名
        output_filename = os.path.join(output_folder, f'overall_visualization_{i}.png')

        # 保存图像
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

        # 关闭当前图像，防止与下一次循环混淆
        plt.clf()
        
        pred_sum = overall.sum()
        print('pred:', pred_sum)
        pred.append(pred_sum)
        targets = np.sum(target)
        print('gt:', targets)
        gt.append(targets)
        
        file.write(f'{img_path}:pred={round(pred_sum)}\tgt={round(targets)}\n')

    mae = mean_absolute_error(pred,gt)
    rmse = np.sqrt(mean_squared_error(pred,gt))
    file.write(f'mae={mae}, rmse={rmse}\n')
    
print ('MAE: ',mae)
print ('RMSE: ',rmse)

