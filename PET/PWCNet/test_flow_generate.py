import os
import subprocess
from multiprocessing import Pool, cpu_count

def process_image_pair(args):
    image_one, image_two, output_flo, model = args
    print(f'Processed {image_one} and {image_two}, output: {output_flo}')
    return subprocess.run(['python', 'run.py', '--model', model, '--one', image_one, '--two', image_two, '--out', output_flo])
    

def generate_flow_pairs(image_dir, output_dir, model='default', num_processes=None):
    # 获取目录中的所有图片文件
    images = sorted([img for img in os.listdir(image_dir) if img.endswith('.png')])
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = []
    for i in range(len(images) - 1):
        image_one = os.path.join(image_dir, images[i])
        image_two = os.path.join(image_dir, images[i + 1])
        output_flo = os.path.join(output_dir, f'flow_{i:04d}.flo')
        args_list.append((image_one, image_two, output_flo, model))
    
    # 如果没有指定进程数，使用CPU核心数
    if num_processes is None:
        num_processes = cpu_count()
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        pool.map(process_image_pair, args_list)

if __name__ == '__main__':
    image_dir = './images/'  # 图片所在目录
    output_dir = './output'  # 输出光流文件目录
    generate_flow_pairs(image_dir, output_dir, num_processes=8)