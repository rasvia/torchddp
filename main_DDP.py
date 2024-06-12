import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from model import MLP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from train_DDP import training_loop


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, init_method='env://')


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    ddp_setup(rank, world_size)

    training_data = datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(training_data, batch_size=64, pin_memory=True, num_workers=4, shuffle=False, sampler=sampler)

    model = MLP(input_dim=28 * 28, hidden_dim=64, output_dim=10).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.MSELoss()
    epoch = 100

    training_loop(model, train_loader, lossFn, optimizer, epoch, rank, world_size)

    cleanup()


if __name__ == '__main__':
    world_size = 4
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size)
