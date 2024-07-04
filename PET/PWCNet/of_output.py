import numpy as np
import cv2

# def load_flow_to_numpy(path):
#     with open(path, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
#         h = np.fromfile(f, np.int32, count=1)[0]
#         w = np.fromfile(f, np.int32, count=1)[0]
#         data = np.fromfile(f, np.float32, count=2 * w * h)
#     data2D = np.resize(data, (w, h, 2))
#     return data2D


# if __name__ == '__main__':
#     flo = load_flow_to_numpy('output1/flow_0000.flo')
#     print(flo.shape)
#     print(flo)

#     import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    image = flow_to_image(flow)
    return image


def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


if __name__ == '__main__':
    image = load_flow_to_png('output1/flow_0000.flo')
    cv2.imwrite('niuniu.png', image)
