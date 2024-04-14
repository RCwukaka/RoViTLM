import os
import sys

import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp

from INet.datasets.PBD import PaderbornBearingDataset
from INet.datasets.TYUT import TYUTDataset
from INet.datasets.WBD import WEBDDataset
from INet.run import config
from INet.training.loss.LRSADTLMLoss import LRSADTLMLoss
from INet.training.output.analysis import getResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import Dataset, DataLoader
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


def run_ddp(rank, world_size, total_epochs, batch_size, lamda, mu, device):
    setup_ddp(rank, world_size)

    for task in config.transfer_task1:
        # run model
        source = WEBDDataset(mapdata=task['source'])
        target = WEBDDataset(mapdata=task['target'])

        datasetReset(source, target)

        for train_model in config.getTrainMode(task['num_class']):

            model = train_model['model']  # load your model
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
            loss = LRSADTLMLoss()

            source_data = prepare_dataloader(source, batch_size)
            target_data = prepare_dataloader(target, batch_size)
            trainer = Trainer(model, source_data, target_data, optimizer, loss, rank, train_model['name'], task['name'],
                              lamda, mu, device)
            trainer.train(total_epochs)

    # for task in config.transfer_task2:
    #     # run model
    #     source = TYUTDataset(mapdata=task['source'])
    #     target = TYUTDataset(mapdata=task['target'])
    #
    #     datasetReset(source, target)
    #
    #     for train_model in config.getTrainMode(task['num_class']):
    #
    #         model = train_model['model']  # load your model
    #         optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    #         loss = LRSADTLMLoss()
    #
    #         source_data = prepare_dataloader(source, batch_size)
    #         target_data = prepare_dataloader(target, batch_size)
    #         trainer = Trainer(model, source_data, target_data, optimizer, loss, rank, train_model['name'], task['name'],
    #                           lamda, mu, device)
    #         trainer.train(total_epochs)

    # for task in config.transfer_task3:
    #     # run model
    #     source = PaderbornBearingDataset(mapdata=task['source'])
    #     target = PaderbornBearingDataset(mapdata=task['target'])
    #
    #     datasetReset(source, target)
    #
    #     for train_model in config.getTrainMode(task['num_class']):
    #
    #         model = train_model['model']  # load your model
    #         optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    #         loss = LRSADTLMLoss()
    #
    #         source_data = prepare_dataloader(source, batch_size)
    #         target_data = prepare_dataloader(target, batch_size)
    #         trainer = Trainer(model, source_data, target_data, optimizer, loss, rank, train_model['name'], task['name'],
    #                           lamda, mu, device)
    #         trainer.train(total_epochs)

    cleanup_ddp()
    # getResult()


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        pin_memory=True,
        shuffle=False,  # 设置了新的 sampler，参数 shuffle 要设置为 False
        sampler=DistributedSampler(dataset)  # 这个 sampler 自动将数据分块后送个各个 GPU，它能避免数据重叠
    )


def datasetReset(source, target):
    if len(source) > len(target):
        source.remove(len(target))
    else:
        target.remove(len(source))


def run_training(world_size, batch_size, total_epochs, lamda, mu, device):
    mp.spawn(run_ddp,
             args=(world_size, total_epochs, batch_size, lamda, mu, device),
             nprocs=world_size,
             join=True)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epochs', type=str, required=False, default=100,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 50')
    parser.add_argument('-world_size', type=int, required=False, default=1,
                        help='[OPTIONAL] Use this flag to specify the number of GPU. Default: 1')
    parser.add_argument('--batch_size', type=int, required=False, default=16,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 32')
    parser.add_argument('--lamda', type=int, required=False, default=1,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 1')
    parser.add_argument('--mu', type=int, required=False, default=5,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 1')
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
                 lamda=args.lamda,
                 mu=args.mu,
                 device=device)


if __name__ == '__main__':
    run_training_entry()
