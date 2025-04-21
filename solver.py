import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import gc
import psutil
import sys
import os
from losses.loss_function import loss_wrapper
from losses.metrics import SDR, cal_SISNR
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
import torchaudio
class Solver(object):
    def __init__(self, args, model, optimizer, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args

        self.loss = loss_wrapper(args.loss_type)

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if not self.args.evaluate_only:
                self.writer = SummaryWriter('%s/tensorboard/' % args.checkpoint_dir)

        self.model = model
        self.optimizer=optimizer
        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.args.local_rank],find_unused_parameters=True)

        if not self.args.evaluate_only:
            self._init()


    def _init(self):
        self.halving = False
        self.step_num = 1
        self.best_val_loss = float("inf")
        self.val_no_impv = 0
        self.start_epoch=1
        self.epoch = 0

        if self.args.train_from_last_checkpoint:
            self._load_model(f'{self.args.checkpoint_dir}/last_best_checkpoint.pt', load_training_stat=True)
        elif self.args.init_from != 'None':
            self._load_model(f'{self.args.init_from}/last_best_checkpoint.pt')
            if self.print: print(f'Init model from {self.args.init_from}, and start new training')
        else:
            if self.print: print('Start new training from scratch')
        self._save_model(self.args.checkpoint_dir+"/last_checkpoint.pt")

        
    def _load_model(self, path, load_optimizer=False, load_training_stat=False):
        checkpoint = torch.load(path, map_location='cpu')

        # load model weights
        pretrained_model = checkpoint['model']
        state = self.model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif f'module.{key}' in pretrained_model and state[key].shape == pretrained_model[f'module.{key}'].shape:
                state[key] = pretrained_model[f'module.{key}']
            elif self.print: print(f'{key} not loaded')
        self.model.load_state_dict(state)

        # load optimizer only
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load the training states
        if load_training_stat:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.step_num = checkpoint['step_num']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']
            self.start_epoch=checkpoint['epoch']
            self.epoch = self.start_epoch-1
            self._print_lr()
            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))

    def _save_model(self, path):
        if self.print:
            checkpoint = {'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch+1,
                            'step_num': self.step_num,
                            'best_val_loss': self.best_val_loss,
                            'val_no_impv': self.val_no_impv}
            torch.save(checkpoint, path)
        
    def _print_lr(self):
        optim_state = self.optimizer.state_dict()
        self.optimizer.load_state_dict(optim_state)
        if self.print: print('Learning rate is: {lr:.6f}'.format(
            lr=optim_state['param_groups'][0]['lr']))

    def _adjust_lr_warmup(self):
        self.warmup_steps = 15000
        if self.step_num < self.warmup_steps:
            lr = self.args.init_learning_rate / 0.001 * (64 ** (-0.5)) * self.step_num * (self.warmup_steps ** (-1.5))
            self.step_num +=1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
   

    def train(self):
        for self.epoch in range(self.start_epoch, self.args.max_epoch+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(self.epoch)
            torch.cuda.empty_cache() 
            # self.check_memory_usage()
            # Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(data_loader = self.train_data)
            if self.args.distributed: tr_loss = self._reduce_tensor(tr_loss)
            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | ''Train Loss {2:.3f}'.format(self.epoch, time.time() - start, tr_loss))
            
            torch.cuda.empty_cache() 
            # self.check_memory_usage()
            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                if self.args.distributed: val_loss = self._reduce_tensor(val_loss)
            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          self.epoch, time.time() - start, val_loss))
            torch.cuda.empty_cache() 
            # self.check_memory_usage()
            # Test
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                if self.args.distributed: test_loss = self._reduce_tensor(test_loss)
            if self.print: print('Test Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Test Loss {2:.3f}'.format(
                          self.epoch, time.time() - start, test_loss))


            # Check whether to early stop and to reduce learning rate
            find_best_model = False
            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv == 5:
                    self.halving = True
                elif self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = val_loss
                find_best_model=True

            # Halfing the learning rate
            if self.halving:
                self.halving = False
                self._load_model(f'{self.args.checkpoint_dir}/last_best_checkpoint.pt', load_optimizer=True)
                if self.print: print('reload weights and optimizer from last best checkpoint')

                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] *= 0.5
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                

            if self.print:
                # Tensorboard logging
                self.writer.add_scalar('Train_loss', tr_loss, self.epoch)
                self.writer.add_scalar('Validation_loss', val_loss, self.epoch)
                self.writer.add_scalar('Test_loss', test_loss, self.epoch)

                self._save_model(self.args.checkpoint_dir+"/last_checkpoint.pt")
                if find_best_model:
                    self._save_model(self.args.checkpoint_dir+"/last_best_checkpoint.pt")
                    print("Fund new best model, dict saved")


    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        # for i, (a_mix, a_tgt, ref_tgt) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
        for i, (a_mix, a_tgt, ref_tgt) in enumerate(data_loader):
            a_mix = a_mix.to(self.args.device)
            a_tgt = a_tgt.to(self.args.device)
            ref_tgt = ref_tgt.to(self.args.device)
            a_tgt_est = self.model(a_mix, ref_tgt)
            
            a_tgt=a_tgt[:, :, 0]
            loss = self.loss(a_tgt, a_tgt_est)

            del a_tgt,a_tgt_est
            gc.collect()

            if state=='train':
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        if self.args.lr_warmup: self._adjust_lr_warmup()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.args.lr_warmup: self._adjust_lr_warmup()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # print(loss)
            total_loss += loss.clone().detach()
           
        return total_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt


    def evaluate(self, data_loader):
        avg_sisnri = 0
        avg_sdri = 0
        avg_pesqi = 0
        avg_stoii = 0

        self._load_model(f'{self.args.checkpoint_dir}/last_best_checkpoint.pt')
        self.model.eval()
        
        with torch.no_grad():
            # for i, (a_mix, a_tgt, ref_tgt) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing Batches"):
            for i, (a_mix, a_tgt, ref_tgt) in enumerate(data_loader):
            
                a_mix = a_mix.to(self.args.device)
                a_tgt = a_tgt.to(self.args.device)
                ref_tgt = ref_tgt.to(self.args.device)
                
                a_tgt_est = self.model(a_mix,ref_tgt)
                a_tgt=a_tgt[:, :, 0]
                a_mix=a_mix[:, :, 0]

                sisnri = cal_SISNR(a_tgt, a_tgt_est) - cal_SISNR(a_tgt, a_mix)
                avg_sisnri += sisnri
          
                a_tgt_est = a_tgt_est.squeeze().cpu().numpy()
                a_tgt = a_tgt.squeeze().cpu().numpy()
                a_mix = a_mix.squeeze().cpu().numpy()

                sdri = SDR(a_tgt, a_tgt_est) - SDR(a_tgt, a_mix)
                avg_sdri += sdri
               
                a_tgt_est = a_tgt_est/np.max(np.abs(a_tgt_est))
                pesqi =  (pesq(self.args.audio_sr, a_tgt, a_tgt_est, 'wb') - pesq(self.args.audio_sr, a_tgt, a_mix, 'wb'))
                avg_pesqi += pesqi
                
                stoii = (stoi(a_tgt, a_tgt_est, self.args.audio_sr, extended=False) - stoi(a_tgt, a_mix, self.args.audio_sr, extended=False))
                avg_stoii += stoii
                
        avg_sisnri = avg_sisnri / (i+1)
        avg_sdri = avg_sdri / (i+1)
        avg_pesqi = avg_pesqi / (i+1)
        avg_stoii = avg_stoii / (i+1)

        print(f'Avg SISNR:i {avg_sisnri}')
        print(f'Avg SNRi: {avg_sdri}')
        print(f'Avg PESQi: {avg_pesqi}')
        print(f'Avg STOIi: {avg_stoii}')


