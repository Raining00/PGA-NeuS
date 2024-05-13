from argparse import ArgumentParser
import torch
import time

def run_on_gpu(memory_size_gb, run_time_seconds):
    # 设置要占用的内存大小（以字节为单位）
    memory_size_bytes = memory_size_gb * 1024**3
    # 创建一个占用指定内存大小的 Tensor
    tensor = torch.empty((memory_size_bytes // 4,), dtype=torch.float, device='cuda')

    # 记录开始时间
    start_time = time.time()

    # 在 GPU 上运行，直到达到指定的运行时间
    while (time.time() - start_time) < run_time_seconds:
        # 在此处执行您希望在 GPU 上运行的操作
        flag = 5
        # if time.time() - start_time > flag:
        #     print("holding gpu for ", time.time() - start_time, " s")
        #     flag = flag + 5
        pass

    # 释放 Tensor
    del tensor

# 调用函数并指定要占用的内存大小（GB）和运行时间（秒）

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu) 
    run_on_gpu(memory_size_gb=9 , run_time_seconds=60000)

"""
python gen.py 
"""