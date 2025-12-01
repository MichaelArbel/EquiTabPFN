import time
import os

from contextlib import nullcontext
import pandas as pd

# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from equitabpfn.patch import setup_environment
from equitabpfn.utils import set_seed
setup_environment()

from torch.cuda.amp import GradScaler
from torch.amp import autocast

from torch import nn
import torch

from mothernet.dataloader import get_dataloader
from mothernet.utils import init_dist, torch_nanmean, check_compatibility

from equitabpfn.evaluation.equivariance_error import average_equivariance_error
from equitabpfn.utils import (
    assign_device,
    get_dtype,
    instantiate,
    Criterion,
    get_original_state_dict
)
from equitabpfn.model_builder import get_model_info, get_model, load_model_from_states

from equitabpfn.eval import Evaluator, Evaluator_OpenML_CC30
from torch.optim.lr_scheduler import SequentialLR

from copy import deepcopy

import equitabpfn.models.equitabpfnv2
# torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True




def eval_prior(prior_config):
    activations = prior_config["mlp"].prior_mlp_activations.choice_values
    activations = [eval(act) for act in activations]
    prior_config["mlp"].prior_mlp_activations.choice_values = activations






class Trainer:
    """ """

    def __init__(self, config, logger):
        """
        Initializes the Trainer class with the provided configuration and logger.

        Parameters:
        - config (object): Configuration object containing various settings.
        - logger (object): Logger object for logging metrics and information.
        """
        self.logger = logger
        self.args = config
        self.device = assign_device(self.args.system.device)
        print(f"Using {self.device} device")
        # self.using_dist, self.rank, self.device = init_dist(self.device)
        self.using_dist = False
        self.rank = 0
        self.dtype = get_dtype(self.args.system.dtype)
        # evaluating module contained in the prior's config
        eval_prior(config["prior"])

        self.dl = get_dataloader(
            prior_config=config["prior"],
            dataloader_config=config["dataloader"],
            device=self.device,
        )
        model_info = get_model_info(config["prior"])
        self.model = get_model(config["model"], model_info)
        if "load" in config:
            self.load_state(config.load)
        self.criterion = Criterion(
            model_info["max_num_classes"], config["model"]["logits"]
        )
        self.model.to(self.device)
        if self.using_dist:
            print("Distributed training")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
            )
        self.dl.model_name = self.model
        check_compatibility(self.dl)
        self.optimizer = instantiate(config["optimizer"]["name"])(
            self.model.parameters(), **config["optimizer"]["kwargs"]
        )

        self.scheduler = self.build_scheduler(config)

        self.scaler = (
            GradScaler() if config["training"]["train_mixed_precision"] else None
        )

        self.config = config["training"]
        # self.evaluator = Evaluator(
        #     self.model,
        #     self.args,
        #     device=self.device,
        #     N_ensemble_configurations=1,
        #     root=self.args.data_path,
        # )
        # self.OpenML_CC30_evaluator = Evaluator_OpenML_CC30(
        #     self.model, self.args, device=self.device, root=self.args.data_path
        # )
        self.epoch = 0


    def build_scheduler(self, config):
        scheduler_1 = instantiate(config["scheduler"]["first"]["name"])(
            self.optimizer, **config["scheduler"]["first"]["kwargs"]
        )
        scheduler_2 = instantiate(config["scheduler"]["second"]["name"])(
            self.optimizer, **config["scheduler"]["second"]["kwargs"]
        )
        scheduler = SequentialLR(
            self.optimizer,
            [scheduler_1, scheduler_2],
            milestones=[config["scheduler"]["warmup_epoch"]],
        )
        return scheduler

    def load_state(self, config):
        path = config["model_state_path"]
        if path:
            states = torch.load(path, map_location="cpu")
            model_state = states[0]

            if not self.load_model_strict:
                for k, v in self.model.state_dict().items():
                    if k in self.model_state and self.model_state[k].shape != v.shape:
                        self.model_state.pop(k)
            self.model.load_state_dict(self.model_state, strict=self.load_model_strict)

    # def small_eval(self):
    #     self.evaluator.set_model(self.model, self.args)
    #     return self.evaluator.eval()

    def save_model(self, name):
        ckpt = (self.model.state_dict(), self.args)
        self.logger.log_artifacts(ckpt, name, artifact_type="torch")

    def save_checkpoint(self):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
        }
        self.logger.log_artifacts(ckpt, "ckpt/last", artifact_type="pickle")

    def load_checkpoint(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]

    def compile_model(self):
        if "compile" in self.args.training:
            if self.args.training.compile:
                self.model = torch.compile(self.model)
            else:
                self.model = self.model

    # def eval(self, ckpt: str = ""):
    #     if ckpt:
    #         try:
    #             print("Loading latest model for evaluation")
    #             states = self.logger.load_artifacts(ckpt, artifact_type="torch")
    #             model, args = load_model_from_states(
    #                 states, device=self.device, verbose=False
    #             )
    #         except:
    #             print(
    #                 "Failed to load checkpoint: file not found. Evaluating current model"
    #             )
    #             model, args = self.model, self.args
    #     else:
    #         print("Evaluating current model")
    #         model, args = self.model, self.args
    #     print(f"Evaluation on test sets")
    #     self.OpenML_CC30_evaluator.set_model(model, args)
    #     openml_results, equivariance_error = self.OpenML_CC30_evaluator.eval()

    #     self.logger.log_artifacts(
    #         openml_results, "test/final_model_OpenML_CC30", artifact_type="pickle"
    #     )
    #     self.logger.log_metrics(
    #         {"equivariance_error": equivariance_error},
    #         "test_final_equivariance",
    #     )
    #     print(openml_results)

    def train(self):
        self.compile_model()
        self.save_checkpoint()
        total_loss = float("inf")
        total_positional_losses = float("inf")
        epochs = self.config["epochs"]
        self.save_model(f"model/epoch_{self.epoch}")
        print(f"Evaluation at epoch {self.epoch}")

        try:
            for epoch in (
                range(self.epoch + 1, epochs + 1)
                if epochs is not None
                else itertools.count(1)
            ):
                epoch_start_time = time.time()
                set_seed(self.args.seed + 10*self.epoch)
                metrics_dict = train_epoch(
                    self.model,
                    self.dl,
                    self.criterion,
                    self.scaler,
                    self.optimizer,
                    self.using_dist,
                    self.device,
                    self.config["aggregate_k_gradients"],
                )

                metrics_dict["lr"] = self.scheduler.get_last_lr()[0]
                metrics_dict["time"] = time.time() - epoch_start_time
                metrics_dict["epoch"] = epoch
                print(pd.DataFrame([metrics_dict]))
                self.logger.log_metrics(metrics_dict, log_name="training")

                if epoch % self.config["ckpt_freq"] == 0:
                    self.save_model(f"model/epoch_{epoch}")
                    self.save_model("model/last")

                self.scheduler.step()
                self.epoch = epoch
                print("Saving checkpoint ...")
                self.save_checkpoint()
                print("New checkpoint saved!")

            self.save_model("model/last")

        except KeyboardInterrupt:
            pass


def train_epoch(
    model,
    dl,
    criterion,
    scaler,
    optimizer,
    using_dist,
    device,
    aggregate_k_gradients,
    # bptt,
    # bptt_extra_samples
):

    model.train()  # Turn on the train mode
    metrics_dict = {
        "total_loss": 0.0,
        # 'total_positional_losses': 0.,
        # 'total_positional_losses_recorded':0,
        "nan_steps": 0,
        "ignore_steps": 0,
    }
    time_before_get_batch = time.time()
    assert (
        len(dl) % aggregate_k_gradients == 0
    ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."


    print(f" loader has size: {len(dl)}")
    step_time = 0
    Offset = 10


    for batch, (data, targets, single_eval_pos) in enumerate(dl):
        # if batch==5:
        #    print(bug)

        if using_dist and not (
            batch % aggregate_k_gradients == aggregate_k_gradients - 1
        ):
            cm = model.no_sync()
        else:
            cm = nullcontext()

        with cm:
            time_to_get_batch = time.time() - time_before_get_batch
            before_forward = time.time()

            inputs = (
                tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                if isinstance(data, tuple)
                else data.to(device)
            )
            new_targets = targets.clone()
            if single_eval_pos is not None:
                new_targets = new_targets[single_eval_pos:]
            new_targets = new_targets.to(device)




            with autocast("cuda", enabled=scaler is not None):
                output = model(inputs, single_eval_pos=single_eval_pos)

                forward_time = time.time() - before_forward


                losses = criterion(output, new_targets)

                losses = losses.view(*output.shape[0:2])
                loss, nan_share = torch_nanmean(losses.mean(0), return_nanshare=True)
                loss = loss / aggregate_k_gradients
            #print(lol)
            if scaler:
                loss = scaler.scale(loss)
            loss.backward()

            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                try:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                except:
                    print("Invalid optimization step encountered")
                optimizer.zero_grad()
            if batch >= Offset:
                ratio = 1./(batch-Offset+1)
                step_time = (1-ratio)*step_time + ratio*(time.time() - before_forward)

                print(f"iter: {batch} in {step_time}")

            if not torch.isnan(loss):
                metrics_dict["total_loss"] += losses.mean().cpu().detach().item()
            metrics_dict["nan_steps"] += nan_share.cpu().item()
            metrics_dict["ignore_steps"] += (new_targets == -100).float().mean().item()

        time_before_get_batch = time.time()

    metrics_dict["total_loss"] /= batch + 1
    metrics_dict["nan_steps"] = metrics_dict["nan_steps"] / (batch + 1)
    metrics_dict["ignore_steps"] /= batch + 1
    metrics_dict.update(
        {
            "step_time": step_time,
            "forward_time": forward_time,
            "time_to_get_batch": time_to_get_batch,
        }
    )

    return metrics_dict


