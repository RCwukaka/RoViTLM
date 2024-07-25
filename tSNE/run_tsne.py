import os
import sys
from os.path import join
from torchsummary import summary
import numpy as np
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.manifold import TSNE
from torch import nn

from RoViTLM.datasets.HUST import HUSTBearingDataset
from RoViTLM.datasets.PBD import PaderbornBearingDataset
from RoViTLM.datasets.WBD import WEBDDataset
from RoViTLM.tSNE import tsne_config

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
from torch.utils.data.distributed import DistributedSampler
current_file_path = join(os.path.abspath(__file__), '../../')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
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
    markers = ['o', 's', 'p', 'H', 'v', '<', '1', '+', '*', 'D', '<', '8', 'x']
    colors = ['#00CED1', '#DC143C', '#e9ccd3', '#ec7696', '#482936', '#ad6598', '#61649f', '#2177b8',
              '#baccd9', '#74787a', '#5bae23', '#485b4d', '#5bae23']

    # for task in tsne_config.transfer_task1:
    #     target = WEBDDataset(mapdata=task['target'])
    #     for train_model in tsne_config.getTrainMode(task['num_class']):
    #         createtSNEandConfusion(batch_size, colors, markers, target, task, train_model)

    for task in tsne_config.transfer_task3:
        target = PaderbornBearingDataset(mapdata=task['target'])
        for train_model in tsne_config.getTrainMode(task['num_class']):

            createtSNEandConfusion(batch_size, colors, markers, target, task, train_model)

    # for task in tsne_config.transfer_task4:
    #     target = HUSTBearingDataset(mapdata=task['target'])
    #     for train_model in tsne_config.getTrainMode(task['num_class']):
    #         createtSNEandConfusion(batch_size, colors, markers, target, task, train_model)

    cleanup_ddp()
    # getResult()


def createtSNEandConfusion(batch_size, colors, markers, target, task, train_model):
    path = join(current_file_path, './training/checkpoint/', train_model['name'], task['name'], './ckpt_50.pth')
    checkpoint = torch.load(path)
    model = nn.DataParallel(train_model['model'])
    model.load_state_dict(checkpoint['net'])
    target_data = prepare_dataloader(target, batch_size)
    output = []
    label = []
    with torch.no_grad():
        for targets in target_data:
            target_output, _, _ = model(targets[0].float(), targets[1].float())
            output.append(target_output)
            label.append(targets[2])
    output = torch.cat(output, dim=0)
    label = torch.cat(label, dim=0)
    output_np = output.view(output.size(0), -1).cpu().detach().numpy()
    label_np = label.view(label.size(0), -1).cpu().detach().numpy()
    X_tsne = TSNE().fit_transform(output_np)
    plt.figure(figsize=(8, 8))
    plt.title(train_model['model_name'])
    pltarr = [0] * task['num_class']
    for index, x_tsne in enumerate(X_tsne):
        t = plt.scatter(x_tsne[0], x_tsne[1], s=200, alpha=0.8, label=np.argmax(label_np[index]), linewidths=0.8,
                        edgecolors=['white'], c=colors[np.argmax(label_np[index])],
                        marker=markers[np.argmax(label_np[index])])
        pltarr[np.argmax(label_np[index])] = t
    plt.legend(pltarr, np.arange(task['num_class']))
    dir = join(current_file_path, './tSNE/output', train_model['name'], task['name'])
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig(dir + '/t-SNE.png', format='png', bbox_inches='tight')
    plt.cla()
    draw_confusion_matrix(np.argmax(label_np, axis=1), np.argmax(output_np, axis=1), title=train_model['model_name'],
                          label_name=np.arange(task['num_class']), pdf_save_path=dir + './confusion.png')


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
    parser.add_argument('--total_epochs', type=str, required=False, default=200,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 50')
    parser.add_argument('-world_size', type=int, required=False, default=1,
                        help='[OPTIONAL] Use this flag to specify the number of GPU. Default: 1')
    parser.add_argument('--batch_size', type=int, required=False, default=16,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 32')
    parser.add_argument('--lamda', type=int, required=False, default=1,
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: 1')
    parser.add_argument('--mu', type=int, required=False, default=1,
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


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    im = plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)
    cbar = add_colorbar(im)
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, format='png', bbox_inches='tight')

    cbar.remove()
    plt.cla()

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

if __name__ == '__main__':
    run_training_entry()
