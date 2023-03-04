import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
from visualdl import LogWriter

from .log import logging_conf, log

import os
import random
import time
import numpy as np
import shutil
import datetime
from tqdm import tqdm


class TrainerConfig(object):

    def __init__(self,
                 name='default',
                 epoch=10,
                 out_dir='./output/',
                 disable_tqdm=False,
                 mixed_precision='no',
                 init_loss_scaling=2.**15,
                 cpu=False,
                 multi_gpu=False,
                 log_with='visualdl',
                 enable_save_checkpoint=True,
                 save_interval=1,
                 save_best=False,
                 save_best_type='min',
                 save_best_rec='val_loss',
                 save_last=True,
                 device=None,
                 sync_bn=False,
                 seed=1024,
                 auto_optimize=True,
                 auto_schedule=True,
                 auto_free=False,
                 # auto_gather_record=True,  # 一次变量重建（清空）后，只能调用一次gather，不然会卡死
                 # auto_clear_record=True,
                 find_unused_parameters=False,
                 ):
        assert save_best_type in ['min', 'max']
        assert log_with in ['visualdl']
        assert mixed_precision in ['no', 'fp16', 'fp16-2']
        assert isinstance(device, str) or isinstance(device, list) or device is None
        self.name = name
        self.epoch = epoch
        self.out_dir = out_dir
        self.disable_tqdm = disable_tqdm
        self.mixed_precision = mixed_precision
        self.init_loss_scaling = init_loss_scaling
        self.cpu = cpu
        self.multi_gpu = multi_gpu
        self.log_with = log_with
        self.enable_save_checkpoint = enable_save_checkpoint
        self.save_interval = save_interval
        self.save_best = save_best
        self.save_best_type = save_best_type
        self.save_best_rec = save_best_rec
        self.save_last = save_last
        self.device = device
        self.sync_bn = sync_bn
        self.seed = seed
        self.auto_optimize = auto_optimize
        self.auto_schedule = auto_schedule
        self.auto_free = auto_free
        self.auto_gather_record = True
        self.auto_clear_record = True
        self.find_unused_parameters = find_unused_parameters


class Trainer(object):

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.model_list = []
        self.optimizer_list = []
        self.lr_scheduler_list = []
        self.component_list = []
        self.records = dict()
        self.save_best_val = -9e9 if self.config.save_best_type == 'max' else 9e9
        self.save_best_calc_func = None
        self.train_global_step = 0
        self.val_global_step = 0
        self.test_global_step = 0
        self.tqdm_state_dict = dict()
        self._init_print()
        self._init_log()
        self._init_checkpoint()
        self._init_runtime()
        self._init_seed()

    def _init_seed(self):
        seed = self.config.seed + paddle.distributed.get_rank()
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def _init_print(self):
        if self.is_local_main_process():
            import logging
            os.makedirs(self.config.out_dir, exist_ok=True)
            # curr_time = datetime.datetime.now()
            # timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
            logging_conf(
                os.path.join(self.config.out_dir, f'{self.config.name}_run.log'),
                level=logging.INFO,
                format='%(asctime)s [INFO] %(message)s',
            )

    def _init_log(self):
        if self.is_local_main_process():
            assert self.config.log_with == 'visualdl'
            self.logger: LogWriter = LogWriter(logdir=os.path.join(self.config.out_dir, 'visualdl_log'))

    def _init_checkpoint(self):
        os.makedirs(os.path.join(self.config.out_dir, 'checkpoint'), exist_ok=True)

    def _init_runtime(self):
        assert self.config.mixed_precision in ['no', 'fp16', 'fp16-2']
        assert isinstance(self.config.device, str) or isinstance(self.config.device, list) or self.config.device is None
        if self.config.device is not None:
            device = self.config.device
            if isinstance(device, list):
                device = ','.join(list(map(lambda x: str(x), device)))
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        if self.config.cpu:
            paddle.set_device('cpu')
        else:
            if self.config.multi_gpu:
                strategy = fleet.DistributedStrategy()
                if self.config.find_unused_parameters:
                    strategy.find_unused_parameters = True
                fleet.init(is_collective=True, strategy=strategy)
            else:
                paddle.set_device('gpu:0')
        if self.config.mixed_precision != 'no':
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.config.init_loss_scaling)
            if self.config.multi_gpu:
                self.scaler = fleet.distributed_scaler(self.scaler)

    def _train_state(self):
        for model in self.get_models():
            model.train()

    def _eval_state(self):
        for model in self.get_models():
            model.eval()

    def _zero_grad(self):
        for optimizer in self.get_optimizers()[0]:
            self.zero_grad(optimizer)

    def _step(self):
        for optimizer in self.get_optimizers()[0]:
            self.step(optimizer=optimizer)

    def _schedule_step(self):
        for lr_schedule in self.get_optimizers()[1]:
            if lr_schedule is not None:
                self.step(lr_scheduler=lr_schedule)

    def _save_checkpoint(self, ep):
        def _del_checkpoint(trainer: Trainer, label, ep_num):
            time.sleep(random.random() * 3)
            if trainer.is_local_main_process():
                for dir_name in os.listdir(os.path.join(trainer.config.out_dir, 'checkpoint')):
                    if dir_name.startswith(label) and not dir_name.startswith(f'{label}_epoch_{ep_num}'):
                        shutil.rmtree(os.path.join(trainer.config.out_dir, 'checkpoint', dir_name))

        def _save_all(trainer: Trainer, checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            for i, model in enumerate(trainer.get_models()):
                paddle.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{i}.pd'))
            for i, optimizer in enumerate(trainer.get_optimizers()[0]):
                paddle.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{i}.pd'))
            for i, lr_schedule in enumerate(trainer.get_optimizers()[1]):
                if lr_schedule is not None:
                    paddle.save(lr_schedule.state_dict(), os.path.join(checkpoint_dir, f'lr_scheduler_{i}.pd'))
            if trainer.config.mixed_precision != 'no':
                paddle.save(trainer.scaler.state_dict(), os.path.join(checkpoint_dir, f'scaler.pd'))

        if ep % self.config.save_interval == 0:
            _save_all(self, os.path.join(self.config.out_dir, 'checkpoint', f'epoch_{ep}'))
        if self.config.save_last:
            _save_all(self, os.path.join(self.config.out_dir, 'checkpoint', f'last_epoch_{ep}'))
            _del_checkpoint(self, 'last', ep)
        if self.config.save_best:
            if self.save_best_calc_func is None:
                rec_dict = self.get_records()
                best_rec = rec_dict[self.config.save_best_rec]
                best_rec = paddle.mean(best_rec).item()
            else:
                best_rec = self.save_best_calc_func(self)
            if (self.config.save_best_type == 'min' and best_rec < self.save_best_val) \
                    or (self.config.save_best_type == 'max' and best_rec > self.save_best_val):
                _save_all(self, os.path.join(self.config.out_dir, 'checkpoint', f'best_epoch_{ep}'))
                self.save_best_val = best_rec
                _del_checkpoint(self, 'best', ep)

    def _update_tqdm_state(self, tqdm_loader, ep, loss):
        self.tqdm_state_dict.update(dict(loss=loss.item() if hasattr(loss, 'item') else loss))
        tqdm_loader.set_description(f'Epoch [{ep}/{self.config.epoch}]')
        tqdm_loader.set_postfix(
            **self.tqdm_state_dict
        )
        self.tqdm_state_dict.clear()

    def _gather_record(self):
        if dist.get_world_size() > 1:
            for k in self.records.keys():
                rst_v = []
                dist.all_gather(rst_v, self.records[k])
                self.records[k] = paddle.concat(rst_v, axis=0)

    def _translate_data_loader(self, data_loader: DataLoader, shuffle):
        if self.config.multi_gpu:
            sampler = DistributedBatchSampler(
                dataset=data_loader.dataset,
                batch_size=data_loader.batch_size,
                shuffle=shuffle,
                drop_last=data_loader.drop_last
            )
            dist_data_loader = DataLoader(
                dataset=data_loader.dataset,
                feed_list=data_loader.feed_list,
                places=None,
                return_list=data_loader.return_list,
                batch_sampler=sampler,
                collate_fn=data_loader.collate_fn,
                num_workers=data_loader.num_workers,
                use_buffer_reader=data_loader.use_buffer_reader,
                use_shared_memory=data_loader.use_shared_memory,
                timeout=data_loader.timeout,
                worker_init_fn=data_loader.worker_init_fn,
                persistent_workers=False,
            )
            return dist_data_loader
        else:
            return data_loader

    # func(trainer: Trainer) -> best_rec
    def set_save_best_calc_func(self, func):
        self.save_best_calc_func = func

    def set_train_dataloader(self, train_loader, shuffle=True):
        self.train_dataloader = self._translate_data_loader(train_loader, shuffle=shuffle)

    def set_val_dataloader(self, val_loader, shuffle=False):
        self.val_dataloader = self._translate_data_loader(val_loader, shuffle=shuffle)

    def set_test_dataloader(self, test_loader, shuffle=False):
        self.test_dataloader = self._translate_data_loader(test_loader, shuffle=shuffle)

    def set_model(self, model):
        self.set_models([model])

    def set_models(self, model_list: list):
        assert isinstance(model_list, list)
        if self.config.sync_bn:
            model_list = [paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model) for model in model_list]
        if self.config.mixed_precision == 'fp16-2':
            self.model_list = paddle.amp.decorate(model_list, level='O2')
        else:
            self.model_list = model_list
        self.model_list = [fleet.distributed_model(model) if self.config.multi_gpu else model for model in self.model_list]

    def get_models(self):
        return self.model_list

    def set_component(self, component):
        self.set_components([component])

    def set_components(self, component_list):
        assert isinstance(component_list, list)
        self.component_list = component_list

    def get_components(self):
        return self.component_list

    def set_optimizer(self, optimizer, lr_scheduler=None):
        self.set_optimizers([optimizer], [lr_scheduler])

    def set_optimizers(self, optimizer_list: list, lr_scheduler_list=None):
        assert isinstance(optimizer_list, list)
        if self.config.mixed_precision == 'fp16-2':
            _, self.optimizer_list = paddle.amp.decorate([], optimizers=optimizer_list, level='O2')
        else:
            self.optimizer_list = optimizer_list
        self.optimizer_list = [fleet.distributed_optimizer(optimizer) if self.config.multi_gpu else optimizer for optimizer in self.optimizer_list]

        self.lr_scheduler_list = []
        if lr_scheduler_list is not None:
            for lr_scheduler in lr_scheduler_list:
                if lr_scheduler is not None:
                    self.lr_scheduler_list.append(
                        lr_scheduler
                    )
                else:
                    self.lr_scheduler_list.append(None)
        if len(self.optimizer_list) > len(self.lr_scheduler_list):
            for _ in range(len(self.optimizer_list) - len(self.lr_scheduler_list)):
                self.lr_scheduler_list.append(None)
        assert len(self.optimizer_list) == len(self.lr_scheduler_list)

    def get_optimizers(self):
        return self.optimizer_list, self.lr_scheduler_list

    def backward(self, loss, **kv):
        if self.config.mixed_precision == 'no':
            loss.backward(**kv)
        else:
            scale_loss = self.scaler.scale(loss)
            scale_loss.backward(**kv)

    def zero_grad(self, optimizer):
        optimizer.clear_grad()

    def step(self, optimizer=None, lr_scheduler=None):
        assert optimizer is not None or lr_scheduler is not None
        if optimizer is not None:
            if self.config.mixed_precision == 'no':
                optimizer.step()
            else:
                self.scaler.step(optimizer)
                self.scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()

    def is_local_main_process(self):
        return dist.get_rank() == 0

    def print(self, txt):
        if self.is_local_main_process():
            log(txt)

    def log(self, value_dict, step):
        if self.is_local_main_process():
            assert isinstance(value_dict, dict)
            for k in value_dict.keys():
                self.logger.add_scalar(k, value_dict[k], step)

    def set_records(self, value_dict):
        assert isinstance(value_dict, dict)
        for k in value_dict.keys():
            v = value_dict[k]
            if not paddle.is_tensor(v):
                v = paddle.to_tensor(v)
            v = paddle.unsqueeze(v, axis=0)
            if k in self.records.keys():
                self.records[k] = paddle.concat([self.records[k], v], axis=0)
            else:
                self.records[k] = v

    def get_records(self):
        return self.records

    def set_bar_state(self, state_dict):
        assert isinstance(state_dict, dict)
        self.tqdm_state_dict.update(state_dict)

    # train_step(trainer: Trainer, bat, bat_idx, global_step) -> loss
    # val_step(trainer: Trainer, bat, bat_idx, global_step) -> loss
    # on_epoch_end(trainer: Trainer, ep) -> None
    def fit(self, train_step, val_step=None, on_epoch_end=None):
        self.train_global_step = 0
        self.val_global_step = 0
        for ep in range(1, self.config.epoch + 1):
            self.print(f'======= epoch {ep} begin ========')
            self._train_state()
            tqdm_loader = tqdm(self.train_dataloader, total=len(self.train_dataloader), disable=not self.is_local_main_process() or self.config.disable_tqdm)
            for bat_idx, bat in enumerate(tqdm_loader):
                if self.config.auto_optimize:
                    self._zero_grad()
                if self.config.mixed_precision == 'no':
                    loss = train_step(self, bat, bat_idx, self.train_global_step)
                else:
                    with paddle.amp.auto_cast(level='O1' if self.config.mixed_precision == 'fp16' else 'O2'):
                        loss = train_step(self, bat, bat_idx, self.train_global_step)
                self.train_global_step += 1
                if self.config.auto_optimize:
                    self.backward(loss)
                    self._step()
                self._update_tqdm_state(tqdm_loader, ep, loss)
            if val_step is not None:
                self._eval_state()
                tqdm_loader = tqdm(self.val_dataloader, total=len(self.val_dataloader), disable=not self.is_local_main_process() or self.config.disable_tqdm)
                for bat_idx, bat in enumerate(tqdm_loader):
                    with paddle.no_grad():
                        if self.config.mixed_precision == 'no':
                            loss = val_step(self, bat, bat_idx, self.val_global_step)
                        else:
                            with paddle.amp.auto_cast(level='O1' if self.config.mixed_precision == 'fp16' else 'O2'):
                                loss = val_step(self, bat, bat_idx, self.val_global_step)
                    self.val_global_step += 1
                    self._update_tqdm_state(tqdm_loader, ep, loss)
            # if self.acc.is_local_main_process:
            if self.config.auto_schedule:
                self._schedule_step()

            if self.config.auto_gather_record:
                self._gather_record()

            if self.config.enable_save_checkpoint and self.is_local_main_process():
                try:
                    time.sleep(random.random() * 5)
                    self._save_checkpoint(ep)
                except:
                    self.print(f'[ERROR] save checkpoint failed : {ep}')
                    pass

            if on_epoch_end is not None:
                on_epoch_end(self, ep)
            if self.config.auto_clear_record:
                self.records.clear()
            if self.config.auto_free:
                paddle.device.cuda.empty_cache()

    # test_step(trainer: Trainer, bat, bat_idx, global_step) -> None
    # on_test_end(trainer: Trainer) -> None
    def test(self, test_step, on_test_end=None):
        self.test_global_step = 0
        self._eval_state()
        tqdm_loader = tqdm(self.test_dataloader, total=len(self.test_dataloader), disable=not self.is_local_main_process() or self.config.disable_tqdm)
        for bat_idx, bat in enumerate(tqdm_loader):
            with paddle.no_grad():
                if self.config.mixed_precision == 'no':
                    test_step(self, bat, bat_idx, self.test_global_step)
                else:
                    with paddle.amp.auto_cast(level='O1' if self.config.mixed_precision == 'fp16' else 'O2'):
                        test_step(self, bat, bat_idx, self.test_global_step)
            self.test_global_step += 1
        if self.config.auto_gather_record:
            self._gather_record()
        if on_test_end is not None:
            on_test_end(self)
        if self.config.auto_clear_record:
            self.records.clear()

    def load_checkpoint(self, checkpoint_dir):
        for i, model in enumerate(self.get_models()):
            model.set_state_dict(paddle.load(os.path.join(checkpoint_dir, f'model_{i}.pd')))
        for i, optimizer in enumerate(self.get_optimizers()[0]):
            optimizer.set_state_dict(paddle.load(os.path.join(checkpoint_dir, f'optimizer_{i}.pd')))
        for i, lr_schedule in enumerate(self.get_optimizers()[1]):
            if lr_schedule is not None:
                lr_schedule.set_state_dict(paddle.load(os.path.join(checkpoint_dir, f'lr_scheduler_{i}.pd')))
        if self.config.mixed_precision != 'no':
            self.scaler.load_state_dict(paddle.load(os.path.join(checkpoint_dir, f'scaler.pd')))


