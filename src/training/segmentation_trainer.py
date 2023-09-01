### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Any, Callable
import pathlib
import time
import math

### External Imports ###
import numpy as np
import torch as tc
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt

### Internal Imports ###
from visualization import volumetric as vol
from evaluation import evaluation_functions as evf
from helpers import utils as u
from helpers import cost_functions as cf

########################



class SegmentationTrainer():
    def __init__(self, **training_params : dict):
        ### General params
        self.model : tc.nn.Module = training_params['model']
        self.device : Union[tc.device, str] = training_params['device']
        self.training_dataloader : tc.utils.data.DataLoader = training_params['training_dataloader']
        self.validation_dataloader : tc.utils.data.DataLoader = training_params['validation_dataloader']
        self.num_iterations : int = training_params['num_iterations']
        self.learning_rate : float = training_params['learning_rate']
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.checkpoint_iters : Iterable[int] = training_params['checkpoint_iters']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        self.logger : Any = training_params['logger']
        self.log_image_iters : Iterable[int]= training_params['log_image_iters']
        self.number_of_images_to_log : int = training_params['number_of_images_to_log']
        self.lr_decay : float = training_params['lr_decay']
        self.inner_batch_size : int = training_params['inner_batch_size']
        self.effective_batch_multiplier : int = training_params['effective_batch_multiplier']
        self.dtype : tc.dtype = training_params['dtype']
        self.log_time : bool = training_params['log_time']
        self.use_amp : bool = training_params['use_amp']
        self.non_blocking : bool = training_params['non_blocking']
        try:
            self.patch_based : bool = training_params['patch_based']
        except:
            self.patch_based : bool = False
        try:
            self.multiclass : bool = training_params['multiclass']
        except:
            self.multiclass : bool = False
        try:
            self.use_sigmoid = training_params['use_sigmoid']
        except:
            self.use_sigmoid = True

        ### Cost functions and params
        self.objective_function : Callable = training_params['objective_function']
        self.objective_function_params : dict = training_params['objective_function_params']
        self.max_gradient_value : float = training_params['max_gradient_value']
        self.max_gradient_norm : float = training_params['max_gradient_norm']

        ### Define optimizers, schedulers and optionally restore checkpoint
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.optimizer = tc.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        self.scheduler = tc.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda epoch: self.lr_decay ** epoch)
        self.load_checkpoint()

    def save_checkpoint(self) -> None:
        to_save = dict()
        to_save['iteration'] = self.current_iter
        to_save['model_state_dict'] = self.model.state_dict()
        to_save['optimizer_state_dict'] = self.optimizer.state_dict()
        to_save['scheduler_state_dict'] =  self.scheduler.state_dict()
        to_save_path = os.path.join(self.checkpoints_path, f"Iteration_{self.current_iter}")
        tc.save(to_save, to_save_path)

    def load_checkpoint(self) -> None:
        if self.to_load_checkpoint_path is not None:
            checkpoint = tc.load(self.to_load_checkpoint_path)
            self.current_iter = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.current_iter = 0

    def initialize_accumulators(self) -> None:
        self.acc_loss = 0.0
        self.acc_dice = 0.0
        self.acc_hausdorff = 0.0
        if self.log_time:
            self.acc_loading_time = 0.0
            self.acc_augmentation_time = 0.0
            self.acc_network_time = 0.0
            self.acc_grad_norm_time = 0.0
            self.acc_optimizer_step_time = 0.0
            self.acc_batch_time = 0.0

    def accumulate_losses(self) -> None:
        self.acc_loss += self.loss.item()
        self.acc_dice += self.dice.item()
        self.acc_hausdorff += self.hausdorff.item()
        if self.log_time:
            try:
                self.acc_loading_time += tc.mean(self.loading_time)
                self.acc_augmentation_time += tc.mean(self.augmentation_time)
            except Exception as e:
                print(f"Error: {e}")
                self.acc_loading_time += self.loading_time
                self.acc_augmentation_time += self.augmentation_time
            self.acc_network_time += self.network_time  
            self.acc_grad_norm_time += self.grad_norm_time
            self.acc_optimizer_step_time += self.optimizer_step_time
            self.acc_batch_time += self.batch_time
        
    def log_losses(self, mode : str) -> None:
        dataset_size = len(self.training_dataloader.dataset) if mode == "training" else len(self.validation_dataloader.dataset)
        self.logger.add_scalar(f"Loss/{mode.capitalize()}/loss", self.acc_loss / dataset_size, self.current_iter)
        self.logger.add_scalar(f"Loss/{mode.capitalize()}/dice", self.acc_dice / dataset_size, self.current_iter)
        self.logger.add_scalar(f"Loss/{mode.capitalize()}/hausdorff", self.acc_hausdorff / dataset_size, self.current_iter)
        if self.log_time:
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_loading", self.acc_loading_time / dataset_size, self.current_iter)
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_augmentation", self.acc_augmentation_time / dataset_size, self.current_iter)
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_network", self.acc_network_time / dataset_size, self.current_iter)
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_grad_norm", self.acc_grad_norm_time / dataset_size, self.current_iter)
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_opt_step", self.acc_optimizer_step_time / dataset_size, self.current_iter)
            self.logger.add_scalar(f"Time/{mode.capitalize()}/time_batch", self.acc_batch_time, self.current_iter)

    def log_images(self, mode : str) -> None:
        dataloader = self.training_dataloader if mode == "training" else self.validation_dataloader
        for (i, batch) in enumerate(dataloader):
            with tc.set_grad_enabled(False):
                self.model.eval()
                if self.patch_based:
                    input_data, ground_truth, spacing = batch[0][0], batch[1][0], batch[2]
                else:
                    input_data, ground_truth, spacing = batch[0], batch[1], batch[2]
                input_data, ground_truth = input_data.type(self.dtype).to(self.device), ground_truth.type(self.dtype).to(self.device)
                output = self.model(input_data)
                if self.use_sigmoid:
                    output = (output > 0.5).type(self.dtype)
                else:
                    output = (tc.nn.Sigmoid()(output) > 0.5).type(self.dtype)
                if tc.sum(output) == 0.0 and tc.sum(ground_truth) == 0.0:
                    dice = 1.0
                elif tc.sum(output) != 0.0 and tc.sum(ground_truth) == 0.0:
                    dice = 0.0
                else:
                    if self.multiclass:
                        dice = (evf.dice_coefficient(output[0, 1, :, :, :].detach().cpu().numpy(), (ground_truth[0, 0, :, :, :]==1).detach().cpu().numpy()) + evf.dice_coefficient(output[0, 2, :, :, :].detach().cpu().numpy(), (ground_truth[0, 0, :, :, :]==2).detach().cpu().numpy())) / 2
                    else:
                        dice = evf.dice_coefficient(output[0, 0, :, :, :].detach().cpu().numpy(), ground_truth[0, 0, :, :, :].detach().cpu().numpy())
                if self.multiclass:
                    buf = vol.show_volumes_2d(input_data[0], ground_truth[0], output[0, 1:2], spacing=spacing[0], return_buffer=True, suptitle=f"Dice: {dice}", names=["Input", "Ground-Truth", "Output"], dpi=100, show=False)
                else:
                    buf = vol.show_volumes_2d(input_data[0], ground_truth[0], output[0], spacing=spacing[0], return_buffer=True, suptitle=f"Dice: {dice}", names=["Input", "Ground-Truth", "Output"], dpi=100, show=False)
                image = PIL.Image.open(buf)
                image = transforms.ToTensor()(image).unsqueeze(0)[0]
                title = f"Mode: {mode.capitalize()} Case: {i}, Iter: {str(self.current_iter)} Dice: {dice}"
                self.logger.add_image(title, image, 0)
                plt.close('all')
            if i >= self.number_of_images_to_log:
                break
            
    def calculate_loss(self, batch : Union[tc.Tensor, tuple], mode : str) -> None:
        gradient_enabled = True if mode == "training" else False
        with tc.set_grad_enabled(gradient_enabled):
            b_t = time.time()
            if self.patch_based:
                input_data, ground_truth = batch[0].type(self.dtype)[0], batch[1].type(self.dtype)[0]
                mult = input_data.shape[0]
            else:
                input_data, ground_truth = batch[0].type(self.dtype), batch[1].type(self.dtype)
                mult = 1
                
            inner_iters = int(math.ceil(input_data.shape[0] / self.inner_batch_size))
            self.dice = 0.0
            self.hausdorff = 0.0
            for i in range(inner_iters):
                bs = self.inner_batch_size
                if i == inner_iters - 1:
                    idata = input_data[i*bs:].to(self.device, non_blocking=self.non_blocking)
                    gtruth = ground_truth[i*bs:].to(self.device, non_blocking=self.non_blocking)
                else:
                    idata = input_data[i*bs:(i+1)*bs].to(self.device, non_blocking=self.non_blocking)
                    gtruth = ground_truth[i*bs:(i+1)*bs].to(self.device, non_blocking=self.non_blocking)
                    
                if self.use_amp and mode == "training":
                    with tc.cuda.amp.autocast(dtype=tc.float16):
                        output = self.model(idata)
                        self.loss = self.objective_function(output, gtruth, **self.objective_function_params) * idata.shape[0] * inner_iters / mult
                else:
                    output = self.model(idata)
                    self.loss = self.objective_function(output, gtruth, **self.objective_function_params) * idata.shape[0] * inner_iters / mult

                if self.multiclass:
                    if self.use_sigmoid:
                        self.dice += cf.dice_loss_monai(output.type(gtruth.dtype), gtruth, to_onehot_y=True, include_background=False) * idata.shape[0] / mult
                        self.hausdorff += cf.hausdorff_loss(output.type(gtruth.dtype), gtruth.long(), to_onehot_y=True, include_background=False) * idata.shape[0] / mult
                    else:
                        self.dice += cf.dice_loss_monai(output.type(gtruth.dtype), gtruth, to_onehot_y=True, include_background=False, sigmoid=True) * idata.shape[0] / mult
                        self.hausdorff += cf.hausdorff_loss(output.type(gtruth.dtype), gtruth.long(), to_onehot_y=True, include_background=False, sigmoid=True) * idata.shape[0] / mult  
                else:
                    if self.use_sigmoid:
                        self.dice += cf.dice_loss(output.type(gtruth.dtype), gtruth) * idata.shape[0] / mult
                        self.hausdorff += cf.hausdorff_loss(output.type(gtruth.dtype), gtruth.long()) * idata.shape[0] / mult
                    else:
                        self.dice += cf.dice_loss(tc.nn.Sigmoid()(output).type(gtruth.dtype), gtruth) * idata.shape[0] / mult
                        self.hausdorff += cf.hausdorff_loss(tc.nn.Sigmoid()(output).type(gtruth.dtype), gtruth.long()) * idata.shape[0] / mult

                if mode == "training":
                    if self.use_amp and mode == "training":
                        self.scaler.scale(self.loss).backward()
                    else:
                        self.loss.backward()
            e_t = time.time()
            self.network_time = e_t - b_t
            if self.log_time:
                total_time = batch[3]
                self.loading_time = total_time[0]
                self.augmentation_time = total_time[1]   

    def get_gradient_norm_and_max(self) -> tuple[float, float]:
        grads = []
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm ** 0.5
        grads = tc.cat(grads)
        return total_norm, tc.max(grads).item()

    def process_batch(self, batch : Union[tc.Tensor, tuple], mode : str) -> None:
        if self.current_batch_pass == 0:
            self.optimizer.zero_grad()

        self.calculate_loss(batch, mode)

        if self.current_batch_pass == self.effective_batch_multiplier - 1:
            if mode == "training":
                b_t = time.time()
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                gradient_norm, gradient_max = self.get_gradient_norm_and_max()
                self.logger.add_scalar(f"Grad/{mode.capitalize()}/grad_max", gradient_max, self.current_iter)
                self.logger.add_scalar(f"Grad/{mode.capitalize()}/grad_norm", gradient_norm, self.current_iter)
                tc.nn.utils.clip_grad_value_(self.model.parameters(), self.max_gradient_value)
                tc.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm, norm_type=2)
                gradient_norm, gradient_max = self.get_gradient_norm_and_max()
                self.logger.add_scalar(f"Grad/{mode.capitalize()}/grad_max_af", gradient_max, self.current_iter)
                self.logger.add_scalar(f"Grad/{mode.capitalize()}/grad_norm_af", gradient_norm, self.current_iter)
                e_t = time.time()
                self.grad_norm_time = e_t - b_t

                b_t = time.time()
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                e_t = time.time()

                self.optimizer_step_time = e_t - b_t
            else:
                self.grad_norm_time = 0.0
                self.current_batch_pass = 0
                self.optimizer_step_time = 0.0
            self.current_batch_pass = 0
        else:
            self.current_batch_pass += 1
            self.grad_norm_time = 0.0
            self.optimizer_step_time = 0.0

    def initialize_amp(self):
        if self.use_amp:
            self.scaler = tc.cuda.amp.GradScaler()
        else:
            pass
            
    def shuffle_dataloaders(self):
        self.training_dataloader.dataset.shuffle()
        self.validation_dataloader.dataset.shuffle()

    def log_iteration_time(self):
        self.logger.add_scalar(f"Time/Iteration_Time", self.iteration_time, self.current_iter)

    def run(self) -> None:
        self.initialize_amp()
        for i in range(self.current_iter, self.num_iterations):
            b_t = time.time()
            tc.cuda.empty_cache()
            self.current_iter = i
            self.current_batch_pass = 0
            ### Training Iteration ###
            self.model.train()
            self.initialize_accumulators()
            print(f"Training, Iter: {self.current_iter}")
            for batch in self.training_dataloader:
                i_b_t = time.time()
                self.process_batch(batch, "training")
                tc.cuda.empty_cache()
                i_e_t = time.time()
                self.batch_time = (i_e_t - i_b_t)
                self.accumulate_losses()
            self.log_losses("training")

            tc.cuda.empty_cache()
            ### Validation Iteration ###
            self.optimizer.zero_grad()
            self.model.eval()
            self.initialize_accumulators()
            print(f"Validation, Iter: {self.current_iter}")
            for batch in self.validation_dataloader:
                i_b_t = time.time()
                self.process_batch(batch, "validation")
                tc.cuda.empty_cache()
                i_e_t = time.time()
                self.batch_time = (i_e_t - i_b_t)
                self.accumulate_losses()
            self.log_losses("validation")
            
            ### Shuffle Dataloaders when iteration size is not default
            self.shuffle_dataloaders()

            ### Update schedulers and (optionally) save the current checkpoint ###
            self.scheduler.step()
            if i in self.checkpoint_iters:
                self.save_checkpoint()
            if i in self.log_image_iters:
                self.log_images("training")
                self.log_images("validation")
            e_t = time.time()
            self.iteration_time = e_t - b_t
            self.log_iteration_time()