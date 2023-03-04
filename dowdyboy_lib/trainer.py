import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from .log import logging_conf, warn

import random
import time
import os
import shutil
import datetime
from tqdm import tqdm

# TODO:
# 2. save best 策略函数优化


class TrainerConfig(object):

    def __init__(self,
                 name='default',
                 epoch=10,
                 out_dir='./output/',
                 disable_tqdm=False,
                 mixed_precision='no',
                 cpu=False,
                 log_with='tensorboard',
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
        assert mixed_precision in ['no', 'fp16', 'bf16']
        assert log_with in ['all', 'tensorboard', 'wandb', 'comet_ml']
        assert isinstance(device, str) or isinstance(device, list) or device is None
        self.name = name
        self.epoch = epoch
        self.out_dir = out_dir
        self.disable_tqdm = disable_tqdm
        self.mixed_precision = mixed_precision
        self.cpu = cpu
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
        self.acc = self._get_acc()
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
        self._init_seed()

    def _init_log(self):
        self.acc.init_trackers(self.config.name, )

    def _init_print(self):
        import logging
        os.makedirs(self.config.out_dir, exist_ok=True)
        # curr_time = datetime.datetime.now()
        # timestamp = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        logging_conf(
            os.path.join(self.config.out_dir, f'{self.config.name}_run.log'),
            level=logging.WARNING,
            format='%(asctime)s [INFO] %(message)s',
            acc=self.acc
        )

    def _init_seed(self):
        set_seed(self.config.seed, True)

    def _init_checkpoint(self):
        os.makedirs(os.path.join(self.config.out_dir, 'checkpoint'), exist_ok=True)

    def _get_acc(self) -> Accelerator:
        assert isinstance(self.config.device, str) or isinstance(self.config.device, list) or self.config.device is None
        if self.config.device is not None:
            device = self.config.device
            if isinstance(device, list):
                device = ','.join(list(map(lambda x: str(x), device)))
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        ddp = DistributedDataParallelKwargs()
        ddp.find_unused_parameters = self.config.find_unused_parameters
        return Accelerator(
            mixed_precision=self.config.mixed_precision,
            cpu=self.config.cpu,
            log_with=self.config.log_with,
            logging_dir=os.path.join(self.config.out_dir, 'acc_log'),
            kwargs_handlers=[ddp],
        )

    def _train_state(self):
        for model in self.get_models():
            model.train()

    def _eval_state(self):
        for model in self.get_models():
            model.eval()

    def _zero_grad(self):
        for optimizer in self.get_optimizers()[0]:
            optimizer.zero_grad()

    def _step(self):
        for optimizer in self.get_optimizers()[0]:
            optimizer.step()

    def _schedule_step(self):
        for lr_schedule in self.get_optimizers()[1]:
            if lr_schedule is not None:
                lr_schedule.step()

    def _save_checkpoint(self, ep):
        def _del_checkpoint(trainer: Trainer, label, ep_num):
            time.sleep(random.random() * 3)
            if trainer.acc.is_local_main_process:
                for dir_name in os.listdir(os.path.join(trainer.config.out_dir, 'checkpoint')):
                    if dir_name.startswith(label) and not dir_name.startswith(f'{label}_epoch_{ep_num}'):
                        shutil.rmtree(os.path.join(trainer.config.out_dir, 'checkpoint', dir_name))
        if ep % self.config.save_interval == 0:
            self.acc.save_state(os.path.join(self.config.out_dir, 'checkpoint', f'epoch_{ep}'))
        if self.config.save_last:
            self.acc.save_state(os.path.join(self.config.out_dir, 'checkpoint', f'last_epoch_{ep}'))
            _del_checkpoint(self, 'last', ep)
        if self.config.save_best:
            if self.save_best_calc_func is None:
                rec_dict = self.get_records()
                best_rec = rec_dict[self.config.save_best_rec]
                best_rec = torch.mean(best_rec).item()
            else:
                best_rec = self.save_best_calc_func(self)
            if (self.config.save_best_type == 'min' and best_rec < self.save_best_val) \
                    or (self.config.save_best_type == 'max' and best_rec > self.save_best_val):
                self.acc.save_state(os.path.join(self.config.out_dir, 'checkpoint', f'best_epoch_{ep}'))
                self.save_best_val = best_rec
                _del_checkpoint(self, 'best', ep)

    def _update_tqdm_state(self, tqdm_loader, ep, loss):
        self.tqdm_state_dict.update(dict(loss=loss.item() if hasattr(loss, 'item') else loss))
        tqdm_loader.set_description(f'Epoch [{ep}/{self.config.epoch}]')
        tqdm_loader.set_postfix(
            **self.tqdm_state_dict
        )
        self.tqdm_state_dict.clear()

    # func(trainer: Trainer) -> best_rec
    def set_save_best_calc_func(self, func):
        self.save_best_calc_func = func

    def set_train_dataloader(self, train_loader):
        self.train_dataloader = self.acc.prepare(train_loader)

    def set_val_dataloader(self, val_loader):
        self.val_dataloader = self.acc.prepare(val_loader)

    def set_test_dataloader(self, test_loader):
        self.test_dataloader = self.acc.prepare(test_loader)

    def set_model(self, model):
        self.set_models([model])

    def set_models(self, model_list: list):
        assert isinstance(model_list, list)
        if self.config.sync_bn:
            model_list = [torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) for model in model_list]
        self.model_list = [self.acc.prepare(model) for model in model_list]

    def get_models(self):
        return self.model_list

    def get_raw_models(self):
        return [self.acc.unwrap_model(model) for model in self.model_list]

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
        self.optimizer_list = [self.acc.prepare(optimizer) for optimizer in optimizer_list]
        self.lr_scheduler_list = []
        if lr_scheduler_list is not None:
            for lr_scheduler in lr_scheduler_list:
                if lr_scheduler is not None:
                    self.lr_scheduler_list.append(
                        self.acc.prepare(lr_scheduler)
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
        self.acc.backward(loss, **kv)

    def zero_grad(self, optimizer):
        optimizer.zero_grad()

    def step(self, optimizer=None, lr_scheduler=None):
        assert optimizer is not None or lr_scheduler is not None
        if optimizer is not None:
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    def device(self):
        return self.acc.device

    def is_local_main_process(self):
        return self.acc.is_local_main_process

    def is_main_process(self):
        return self.acc.is_main_process

    def print(self, txt):
        warn(txt, acc=self.acc)

    def log(self, value_dict, step):
        assert isinstance(value_dict, dict)
        self.acc.log(value_dict, step)

    def set_records(self, value_dict):
        assert isinstance(value_dict, dict)
        for k in value_dict.keys():
            v = value_dict[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=self.acc.device)
            else:
                v = v.to(self.acc.device)
            v = torch.unsqueeze(v, dim=0)
            if k in self.records.keys():
                self.records[k] = torch.cat([self.records[k], v], dim=0)
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
            tqdm_loader = tqdm(self.train_dataloader, total=len(self.train_dataloader), disable=not self.acc.is_local_main_process or self.config.disable_tqdm)
            for bat_idx, bat in enumerate(tqdm_loader):
                if self.config.auto_optimize:
                    self._zero_grad()
                loss = train_step(self, bat, bat_idx, self.train_global_step)
                self.train_global_step += 1
                if self.config.auto_optimize:
                    self.acc.backward(loss)
                    self._step()
                self._update_tqdm_state(tqdm_loader, ep, loss)
            if val_step is not None:
                self._eval_state()
                tqdm_loader = tqdm(self.val_dataloader, total=len(self.val_dataloader), disable=not self.acc.is_local_main_process or self.config.disable_tqdm)
                for bat_idx, bat in enumerate(tqdm_loader):
                    with torch.no_grad():
                        loss = val_step(self, bat, bat_idx, self.val_global_step)
                        self.val_global_step += 1
                    self._update_tqdm_state(tqdm_loader, ep, loss)
            # if self.acc.is_local_main_process:
            if self.config.auto_schedule:
                self._schedule_step()

            if self.config.auto_gather_record:
                self.records = self.acc.gather(self.records)

            if self.config.enable_save_checkpoint:
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
                self.acc.free_memory()

    # test_step(trainer: Trainer, bat, bat_idx, global_step) -> None
    # on_test_end(trainer: Trainer) -> None
    def test(self, test_step, on_test_end=None):
        self.test_global_step = 0
        self._eval_state()
        tqdm_loader = tqdm(self.test_dataloader, total=len(self.test_dataloader), disable=not self.acc.is_local_main_process or self.config.disable_tqdm)
        for bat_idx, bat in enumerate(tqdm_loader):
            with torch.no_grad():
                test_step(self, bat, bat_idx, self.test_global_step)
                self.test_global_step += 1
        if self.config.auto_gather_record:
            self.records = self.acc.gather(self.records)
        if on_test_end is not None:
            on_test_end(self)
        if self.config.auto_clear_record:
            self.records.clear()

    def load_checkpoint(self, checkpoint_dir):
        self.acc.load_state(checkpoint_dir)


