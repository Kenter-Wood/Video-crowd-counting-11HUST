import os
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy

def extract_points_from_json(json_file_path):
    """
    从 JSON 文件中提取点信息
    :param json_file_path: JSON 文件路径
    :return: 点信息列表 [(x1, y1), (x2, y2), ...]
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        points = []
        for shape in data.get('shapes', []):
            for point in shape.get('points', []):
                points.append((point[0], point[1]))
        return points

def extract_points_from_directory(directory_path):
    """
    提取目录下所有 JSON 文件中的点信息并生成字典
    :param directory_path: 目录路径
    :return: 字典 {filename: [(x1, y1), (x2, y2), ...]}
    """
    points_dict = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory_path, filename)
            points = extract_points_from_json(json_file_path)
            img_name = filename.split('.')[0] + '.jpg'
            points_dict[img_name] = points
    return points_dict


def generate_density_map(points, image):
    """
    生成密度图
    :param points: 标注点的坐标列表 [(x1, y1), (x2, y2), ...]
    :param map_size: 输出密度图的大小 (width, height)
    :param sigma: 高斯核的标准差
    :return: 生成的密度图
    """
    h, w = image.shape[:, 2]
    density_map = np.zeros((h, w), dtype=np.float32)

    if len(points) == 1:
        x1 = max(1, min(w, round(points[0, 0])))
        y1 = max(1, min(h, round(points[0, 1])))
        density_map[y1, x1] = 255
        return density_map

    #建立搜索树
    pts = np.array(zip(np.nonzero(image)[1], np.nonzero(image)[0]))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    # 对每个标注点开始遍历
    # 定义一个高斯窗口的尺寸(f_sz)和标准差(sigma)，并使用fspecial函数创建一个高斯滤波器(H)
    for point in points:
        f_sz = 15
        # kd数搜索出离该目标点最近的4个标注点距离，去除最近的，剩余3个距离相加取平均，再乘上文章中取的beta=0.3
        sigma = (distances[point][1] + distances[point][2] + distances[point][3]) * 0.3 / 3
        # 生成高斯核
        H = gaussian_filter(np.zeros((f_sz, f_sz)), sigma)
        # 将坐标点坐标floor向下取整，计算出在矩阵中的位置，并限定在图像最大尺寸与【1，1】之间
        x = min(w, max(1, int(np.floor(point[0]))))
        y = min(h, max(1, int(np.floor(point[1]))))
        # 超出范围的点舍弃
        if x > w or y > h:
            continue
        # 根据窗口大小计算出左上角(x1,y1)和右下角坐标(x2,y2)
        x1 = x - int(f_sz / 2)
        y1 = y - int(f_sz / 2)
        x2 = x + int(f_sz / 2)
        y2 = y + int(f_sz / 2)

        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False
        # 判断高斯核尺寸是否超过图像边界如果超过则改变
        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        # 计算裁剪后的区域在高斯滤波器核矩阵中的宽度和高度
        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2

        if change_H:
            H = gaussian_filter(np.zeros((y2h - y1h + 1, x2h - x1h + 1)), sigma)
        # 将高斯滤波器(H)加到矩阵 im_density 的相应位置上，以构建密度图
        density_map[y1 - 1:y2, x1 - 1:x2] += H

    return density_map


if __name__ == '__main__':
    # 配置参数
    directory_path = 'test_json'  # 请替换为你的 JSON 文件目录路径

    # 提取点信息并生成字典
    points_dict = extract_points_from_directory(directory_path)
    # print(points_dict.keys())

    density_maps = {}
    for img_path in points_dict.keys():
        points = points_dict[img_path]
        path = 'data/WuhanMetro/test_data/images/' + img_path
        image = cv2.imread(path)
        density_map = generate_density_map(points, image)
        density_maps[img_path] = density_map

    img_list = list(density_maps.keys())

    image_path = 'data/WuhanMetro/test_data/images/' + img_list[150]
    image = cv2.imread(image_path)
    dstmp = density_maps[img_list[150]]

    cv2.imshow('image', image)
    cv2.imshow('dstmp', dstmp)
    cv2.waitKey(0)