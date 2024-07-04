import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if 202021.25 != magic:
            print(f'{filename} Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = int(np.fromfile(f, np.int32, count=1)[0])
            h = int(np.fromfile(f, np.int32, count=1)[0])
            data = np.fromfile(f, np.float32, count=2 * w * h)
            return np.resize(data, (h, w, 2))

def visualize_flow_custom(flow, output_path):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    gray = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    gray = np.uint8(gray)

    cv2.imwrite(output_path, gray)

# 获取所有 .flo 文件
input_folder = 'output5'
flo_files = [f for f in os.listdir(input_folder) if f.endswith('.flo')]

# 批量处理 .flo 文件
for flo_file in flo_files:
    flow = read_flo_file(os.path.join(input_folder, flo_file))
    if flow is not None:
        output_file = os.path.join('vis5', f'{os.path.splitext(flo_file)[0]}.png')
        visualize_flow_custom(flow, output_file)
        print(f"Flow visualization saved to {output_file}")
