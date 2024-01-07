import os
import sys
from os.path import join
from typing import List

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from time import time, sleep

from tqdm import tqdm

from INet.training.logger.net_logger import NetLogger
from torch import distributed as dist, autocast

from INet.training.lr_scheduler.polylr import PolyLRScheduler
from INet.utilities.collate_outputs import collate_outputs
from INet.utilities.helpers import dummy_context, empty_cache


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            source_data: DataLoader,
            target_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            loss: torch.nn.Module,
            gpu_id: int,
            model_name: str,
            lamda: int,
            device: torch.device = torch.device('cuda'),
    ) -> None:
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.num_epochs = 1000
        self.lamda = lamda
        self.model_name = model_name

        self.current_epoch = 0
        self.train_outputs = {}
        self.gpu_id = gpu_id
        self.device = device
        self.source_data = source_data
        self.target_data = target_data
        self.optimizer = optimizer
        self.lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        self.loss = loss
        self.model = DDP(model.to(self.device), device_ids=[gpu_id])
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        self.logger = NetLogger()
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        current_file = __file__
        self.absolute_path = os.path.abspath(current_file)
        self.output_folder = join(self.absolute_path, '../fold_logging')
        timestamp = datetime.now()
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.checkpointpath = join(self.absolute_path, '../checkpoint/' + self.model_name)
        self.RESUME = False

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)
        if self.RESUME:
            path_checkpoint = self.checkpointpath
            dirList = os.listdir(path_checkpoint)
            checkpoint = torch.load(join(path_checkpoint, dirList[0]))
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_schedule'])

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file(
            f"train_loss: {np.round(self.logger.my_fantastic_logging['train_loss'][-1], decimals=4)} "
            f"train_accuracy: {np.round(self.logger.my_fantastic_logging['train_accuracy'][-1], decimals=4)} ")
        # self.print_to_log_file(
        #     f"target_loss: {np.round(self.logger.my_fantastic_logging['target_loss'][-1], decimals=4)} "
        #     f"target_accuracy: {np.round(self.logger.my_fantastic_logging['target_accuracy'][-1], decimals=4)} ")
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    def on_train_start(self):
        if self.is_ddp:
            dist.barrier()

    def on_train_end(self):
        empty_cache(self.device)
        self.print_to_log_file("Training done.")


    def on_train_epoch_start(self):
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch + 1}/{self.num_epochs}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            train_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(train_tr, outputs)
            train_loss = []
            train_accuracy = []
            for _ in train_tr:
                train_loss = np.append(train_loss, _['train_loss'])
                train_accuracy = np.append(train_accuracy, _['train_accuracy'])
            train_loss_here = train_loss.mean()
            train_accuracy_here = train_accuracy.mean()
        else:
            train_loss_here = np.mean(outputs['train_loss'])
            train_accuracy_here = np.mean(outputs['train_accuracy'])

        self.logger.log('train_loss', train_loss_here, self.current_epoch)
        self.logger.log('train_accuracy', train_accuracy_here, self.current_epoch)
        self.current_epoch += 1
        if self.current_epoch > int(self.num_epochs / 2) and self.current_epoch % 3 == 0:
            self.savePoint()

    def on_target_epoch_start(self):
        return

    def on_target_epoch_end(self, target_outputs: List[dict]):
        outputs = collate_outputs(target_outputs)

        if self.is_ddp:
            val_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(val_tr, outputs)
            target_loss = []
            target_accuracy = []
            for _ in val_tr:
                target_loss = np.append(target_loss, _['target_loss'])
                target_accuracy = np.append(target_accuracy, _['target_accuracy'])
            target_loss_here = target_loss.mean()
            target_accuracy_here = target_accuracy.mean()
        else:
            target_loss_here = np.mean(outputs['target_loss'])
            target_accuracy_here = np.mean(outputs['target_accuracy'])

        self.logger.log('target_loss', target_loss_here, self.current_epoch)
        self.logger.log('target_accuracy', target_accuracy_here, self.current_epoch)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            # while not successful and ctr < max_attempts:
            #     try:
            #         with open(self.log_file, 'a+') as f:
            #             for a in args:
            #                 f.write(str(a))
            #                 f.write(" ")
            #             f.write("\n")
            #         successful = True
            #     except IOError:
            #         print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
            #         sleep(0.5)
            #         ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def savePoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_schedule': self.lr_scheduler.state_dict()}
        dir = join(self.absolute_path, '../checkpoint/' + self.model_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        torch.save(checkpoint, dir + '/ckpt_%s.pth' % (str(self.current_epoch)))

    def train_step(self):
        self.source_data.sampler.set_epoch(self.current_epoch)
        train_outputs = []
        loop = tqdm(total=len(self.source_data), file=sys.stdout)
        for sources, targets in zip(self.source_data, self.target_data):
            loop.update(1)
            sources_X1 = sources[0].to(self.gpu_id)
            sources_X2 = sources[1].to(self.gpu_id)
            sources_label = sources[2].to(self.gpu_id)
            targets_X1 = targets[0].to(self.gpu_id)
            targets_X2 = targets[1].to(self.gpu_id)
            targets_label = targets[2].to(self.gpu_id)
            train_outputs.append(self._run_batch(sources_X1.float(), sources_X2.float(),
                                                 sources_label.float(), targets_X1.float(),
                                                 targets_X2.float(), targets_label.float()))
        loop.close()
        return train_outputs

    def _run_batch(self, sources_X1, sources_X2, sources_label, targets_X1, targets_X2, targets_label):
        batch_size = len(sources_X1)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            source_output, source_feature, \
            target_output, target_feature, \
            source_domain_output, target_domain_output = self.model(sources_X1, sources_X2, targets_X1, targets_X2)
            l = self.loss(source_output, sources_label, source_feature,
                          target_feature, source_domain_output, target_domain_output,
                          lamda=self.lamda)
            predicted = torch.argmax(target_output, 1)
            accuracy = (predicted == torch.argmax(targets_label, 1)).sum().item()
            del sources_X1, sources_X2, sources_label, targets_X1, targets_X2, targets_label

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
            self.optimizer.step()

        return {'train_loss': l.detach().cpu().numpy(), 'train_accuracy': accuracy / batch_size}

    def target_step(self):
        self.target_data.sampler.set_epoch(self.current_epoch)
        self.model.eval()
        with torch.no_grad():
            target_outputs = []
            for source1, source2, label in self.target_data:
                source1 = source1.to(self.gpu_id)
                source2 = source2.to(self.gpu_id)
                label = label.to(self.gpu_id)
                class_output, domain_output = self.model(source1.float(), source2.float())
                predicted = torch.argmax(class_output, 1)
                accuracy = (predicted == torch.argmax(label, 1)).sum().item()
                del source1, source2
                l = self.loss(class_output, domain_output, label.float(),
                              domain_label=np.tile([1., 0.], [len(label), 1]))
                target_outputs.append(
                    {'target_loss': l.detach().cpu().numpy(), 'target_accuracy': accuracy / class_output.size(0)})
            return target_outputs

    def train(self, max_epochs: int):
        self.on_train_start()
        self.num_epochs = max_epochs
        for epoch in range(max_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = self.train_step()
            self.on_train_epoch_end(train_outputs)

            # self.on_target_epoch_start()
            # target_outputs = self.target_step()
            # self.on_target_epoch_end(target_outputs)

            self.on_epoch_end()
            if self.current_epoch + 1 >= max_epochs:
                break
        self.on_train_end()
