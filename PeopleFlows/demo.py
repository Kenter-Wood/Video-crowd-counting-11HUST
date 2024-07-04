import cv2
import os
import numpy as np
from scipy.ndimage import maximum_filter, label

def find_local_maxima(image, size=10):
    # 使用最大滤波器找到局部极大值
    max_filtered = maximum_filter(image, size=size)
    local_maxima = (image == max_filtered)
    labeled, num_objects = label(local_maxima)
    return local_maxima, labeled, num_objects

# 文件夹路径
folder_path = 'vis'

# 获取文件夹内所有的jpg文件
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

for file_name in jpg_files:
    # 读取图像
    file_path = os.path.join(folder_path, file_name)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"图像 {file_name} 读取失败")
        continue
    
    # 找到局部极大值点
    local_maxima, labeled, num_objects = find_local_maxima(image)
    
    # 打印或处理局部极大值点
    print(f"{file_name} 中找到 {num_objects} 个局部极大值点")
    # 你可以在此处添加更多的处理逻辑，例如绘制局部极大值点

    # 示例：绘制局部极大值点
    image_with_maxima = image.copy()
    image_with_maxima[local_maxima] = 255
    cv2.imwrite(f'./prcsdimg/{file_name}', image_with_maxima)

