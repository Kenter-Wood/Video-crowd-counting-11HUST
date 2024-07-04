import torch.utils.data
import torchvision
from .FRVCCdata import FRVCCLoader

from .SHA import build as build

data_path = {
    'SHA': './data/ShanghaiTech/part_A',
    'MyData': './data/MyData',
    'HT21': './data/HT21',
    'Xuezhang': './Xuezhang_WM',
    'WuhanMetro': './data/WuhanMetro_origin'
}


def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build(image_set, args)
    elif args.dataset_file == 'MyData':
        return build(image_set, args)
    elif args.dataset_file == 'HT21':
        return build(image_set, args)
    elif args.dataset_file == 'Xuezhang':
        return build(image_set, args)
    elif args.dataset_file == 'WuhanMetro':
        return build(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_FRVCCLoader(img_folder, density_folder, gt_folder):
    return FRVCCLoader(img_folder, density_folder, gt_folder)
