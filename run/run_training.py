import os
import sys

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as Data

from INet.training.loss.LRSADTLMLoss import LRSADTLMLoss
from INet.training.model.LRSADTLM.LRSADTLM import LRSADTLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import Dataset, DataLoader
from INet.datasets.TYUT import TYUTDataset
from INet.training.Trainer import Trainer

from torch.utils.data.distributed import DistributedSampler


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(rank, world_size, total_epochs, batch_size, test_size, model_name, device):
    setup_ddp(rank, world_size)

    # run model
    source, target, model, optimizer, loss = load_train_objs()
    # train_dataset, test_dataset = split_dataset(data, test_size)
    source_data = prepare_dataloader(source, batch_size)
    target_data = prepare_dataloader(target, batch_size)
    trainer = Trainer(model, source_data, target_data, optimizer, loss, rank, model_name, device)
    trainer.train(total_epochs)

    cleanup_ddp()


# def split_dataset(data: Dataset, test_size: int):
#     test_size = int(len(data) * test_size)
#     train_size = len(data) - test_size
#     train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
#     return train_dataset, test_dataset


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        pin_memory=True,
        shuffle=False,  # 设置了新的 sampler，参数 shuffle 要设置为 False
        sampler=DistributedSampler(dataset)  # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
    )


def load_train_objs():
    source = TYUTDataset()  # load your datasets
    target = TYUTDataset()  # load your datasets
    model = LRSADTLM()  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = LRSADTLMLoss()
    return source, target, model, optimizer, loss


def run_training(world_size, batch_size, total_epochs, test_size, model_name, device):
    mp.spawn(run_ddp,
             args=(world_size, total_epochs, batch_size, test_size, model_name, device),
             nprocs=world_size,
             join=True)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epochs', type=str, required=False, default=50,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 50')
    parser.add_argument('-world_size', type=int, required=False, default=1,
                        help='[OPTIONAL] Use this flag to specify the number of GPU. Default: 1')
    parser.add_argument('--batch_size', type=int, required=False, default=32,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 32')
    parser.add_argument('--test_size', type=int, required=False, default=0.2,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 0.2')
    parser.add_argument('-device', type=str, default='cuda', required=False)
    args = parser.parse_args()

    assert args.device in ['cpu',
                           'cuda'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'

    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    else:
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')

    run_training(world_size=args.world_size,
                 batch_size=args.batch_size,
                 total_epochs=args.total_epochs,
                 test_size=args.test_size,
                 model_name="net",
                 device=device)


if __name__ == '__main__':
    run_training_entry()
