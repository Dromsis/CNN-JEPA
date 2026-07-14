# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import time
from typing import Callable
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
try:
    # pytorch-lightning >= 1.9 renamed lightning_lite -> lightning_fabric
    from lightning_fabric.utilities.rank_zero import _get_rank
except Exception:
    try:
        from lightning_lite.utilities.rank_zero import _get_rank  # pl 1.8
    except Exception:
        # Last resort: derive rank from the launcher env (single-GPU -> 0).
        def _get_rank():
            for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
                if k in os.environ:
                    return int(os.environ[k])
            return 0
import torch
import torchvision
import torch.nn as nn
import timm
import wandb
import matplotlib.pyplot as plt

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule, CosineWarmupScheduler
from lightly.utils.lars import LARS

from pretrain.metrics import contrastive_acc_eval, log_example_inputs, eval_feature_descriptors
from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
import utils

from data.hdf5_imagefolder import HDF5ImageFolder
from data.flat_image_folder import FlatImageFolder

class LightlyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()  # save cfg to self.hparams
        self.cfg = cfg
        self.lr = cfg.optimizer.lr

        self.backbone = timm.create_model(
            cfg.backbone.name,
            pretrained=cfg.backbone.pretrained_weights == "imagenet",
            num_classes=0,
            **dict(cfg.backbone.kwargs),
        )
        # This sets which backbone to use for online eval. By default, it's the same as the main backbone
        # Override this if needed (e.g. with backbone_momentum)
        self.backbone_for_online_eval = self.backbone 

        self.projection_head = None
        self.criterion = None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.cfg.get("reset_lr_scheduler", False):
            if "lr_schedulers" in checkpoint:
                print("Resetting lr_schedulers from checkpoint to empty list to bypass state restoration.", flush=True)
                checkpoint["lr_schedulers"] = []


    def forward(self, x):

        """Implment forward step for each method!

        Args:
            x: a minibatch of augmented input images
        """
        raise NotImplemented

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        """Implment train_val step for each method!
        log loss using self.log(f"{metric_label}/loss", loss, on_epoch=True)
        """
        raise NotImplemented

    def setup_transform(self):
        """ Set sef.transform to a lightly transform by oveerriding this method.
            Use sef.input_size to set the input_size of the transform.
        """
        # We set self.transform to an invalid value to allow this function to be called, but if it's not overriden, we raise an error
        # self.setup calls this method, therefore this hack to allows this class to be instantiated without having to override this method
        self.transform = -1

    def training_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx)
        self.log(f"train_metrics/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(f"train_metrics/wd", self.trainer.optimizers[0].param_groups[0]["weight_decay"])
        
        if hasattr(self, "wd_scheduler"):
            self.wd_scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx, metric_label="val_metrics")
        if self.trainer.sanity_checking:
            views = self.get_views_to_log_from_batch(batch)
            shuffle = torch.randperm(views[0].shape[0])
            views = [view[shuffle] for view in views]
            log_example_inputs(views, log_label="val")
        return loss

    def get_views_to_log_from_batch(self, batch):
        # a batch in lightly is a tuple: inputs, targets, filepaths. Views are in batch[0]
        # Override this if the transforms doewsn't return multiple views in inputs
        return batch[0]

    def on_validation_epoch_end(self) -> None:
        # The online linear-probe benchmark needs labels and scans the whole train+val set to
        # fit a classifier every few epochs. With unlabeled data it is meaningless AND very
        # slow, so it can be turned off with data.online_benchmark=false.
        if not self.cfg.data.get("online_benchmark", True):
            return
        if not self.trainer.sanity_checking:
            if self.current_epoch % 5 == 0:
                try:
                    benchmark_results_dict = self.online_classifier.run_benchmarks(
                        device=self.device,
                        dist_all_gather_fcn=self.all_gather,
                        train_dataloader=self.trainer.train_dataloader,
                        val_dataloader=self.trainer.val_dataloaders[0],
                        train_val_transform=self.transform,
                    )
                except Exception as e:
                    print(f"Failed to run online classification benchmarks: {e}", flush=True)
                    benchmark_results_dict = {"lin_top1": 0.0}

                # if benchmark_results_dict is not None:
                self.log_dict({f"val_metrics/{k}": v for k, v in benchmark_results_dict.items()})
                # https://github.com/Lightning-AI/pytorch-lightning/issues/19045

    def configure_optimizers(self):
        if self.cfg.optimizer.get('exclude_norm_and_bias_from_wd', False):
            params, params_no_weight_decay, _, param_names_no_weight_decay = utils.get_weight_decay_parameters(self.named_parameters())
            print("Parameters excluded from weight decay:", param_names_no_weight_decay, flush=True)
            param_groups = [
                {
                    'params': params
                }, 
                {
                    'params': params_no_weight_decay,
                    'WD_exclude': True, # important for CosineWDSchedule
                    'weight_decay': 0
                }
            ]
        else:
            param_groups = self.parameters()

        wd = self.cfg.optimizer.get('weight_decay', 0.0)
        if self.cfg.optimizer.get('algorithm', 'adamw').lower() == 'lars':
            optim = LARS(
                param_groups,
                lr=self.cfg.optimizer.lr,
                momentum=0.9,
                weight_decay=wd,
            )
        else:        
            optim = torch.optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=wd,
            )

        if self.cfg.optimizer.get('wd_schedule', False): 
            self.wd_scheduler = utils.CosineWDSchedule(
                optim,
                ref_wd=wd,
                final_wd=wd * 10,
                T_max=int(self.trainer.estimated_stepping_batches),
            )

        if self.cfg.optimizer.get('cosine_warmpup_sched', False):
            if self.cfg.get("reset_lr_scheduler", False):
                # If we reset the scheduler, calculate remaining epochs and steps to decay properly to 0
                current_epoch = self.current_epoch
                max_epochs = self.cfg.trainer.max_epochs
                batches_per_epoch = int(self.trainer.estimated_stepping_batches / max_epochs)
                remaining_epochs = max_epochs - current_epoch
                remaining_steps = max(1, remaining_epochs * batches_per_epoch)
                
                warmup_epochs = 0
                max_steps = remaining_steps
                print(f"Resumed scheduler: current_epoch={current_epoch}, max_epochs={max_epochs}, batches_per_epoch={batches_per_epoch}, remaining_steps={remaining_steps}. Scheduler will decay from {self.lr} over {remaining_steps} steps without warmup.", flush=True)
            else:
                warmup_epochs = int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * self.cfg.optimizer.get('lr_warmup_epochs', 10)
                )
                max_steps = int(self.trainer.estimated_stepping_batches)

            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optim,
                    warmup_epochs=warmup_epochs,
                    max_epochs=max_steps,
                ),
                "interval": "step",
            }
            return [optim], [scheduler]
        else:
            return optim

    def setup(self, stage: str) -> None:
        dataset_classes = {
            "maritime": HDF5ImageFolder, # flat, label-free dir of images packaged in HDF5 format
        }
        import os
        custom_train = self.cfg.data.get("train_hdf5_path", None)
        custom_val = self.cfg.data.get("val_hdf5_path", None)

        if custom_train and os.path.exists(custom_train):
            maritime_train_path = custom_train
        elif os.path.exists("/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-train.h5"):
            maritime_train_path = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-train.h5"
        elif os.path.exists("/home/dromsis/Pictures/dataset/combined/maritime-train.h5"):
            maritime_train_path = "/home/dromsis/Pictures/dataset/combined/maritime-train.h5"
        else:
            maritime_train_path = "/data/maritime-train.h5"

        if custom_val and os.path.exists(custom_val):
            maritime_val_path = custom_val
        elif os.path.exists("/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-val.h5"):
            maritime_val_path = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-val.h5"
        elif os.path.exists("/home/dromsis/Pictures/dataset/combined/maritime-val.h5"):
            maritime_val_path = "/home/dromsis/Pictures/dataset/combined/maritime-val.h5"
        else:
            maritime_val_path = "/data/maritime-val.h5"

        train_dataset_kwargs = {
            "maritime": dict(root=maritime_train_path, subsample=self.cfg.data.get("subsample", 1)),
        }
        val_dataset_kwargs = {
            "maritime": dict(root=maritime_val_path, subsample=self.cfg.data.get("subsample", 1)),
        }
        input_sizes = {
            "maritime": 640,
        }
        num_classes = {
            "maritime": 1, # dummy; data is unlabeled (FlatImageFolder returns label 0)
        }
        self.dataset_class = dataset_classes[self.cfg.data.dataset_name]
        self.train_dataset_kwargs = train_dataset_kwargs[self.cfg.data.dataset_name]
        self.val_dataset_kwargs = val_dataset_kwargs[self.cfg.data.dataset_name]
        self.input_size = input_sizes[self.cfg.data.dataset_name]
        self.num_classes = num_classes[self.cfg.data.dataset_name]

        # Setup self.transform
        self.setup_transform()

        self.train_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.train_dataset_kwargs),
            transform=self.transform
        )
        self.val_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.val_dataset_kwargs),
            transform=self.transform
        )

        lin_benchmark_train_kwargs = self.train_dataset_kwargs.copy()
        self.online_classifier = OnlineLinearClassificationBenckmark(
            backbone=self.backbone_for_online_eval,
            num_classes=self.num_classes, 
            dataset_class=self.dataset_class,
            train_dataset_kwargs = lin_benchmark_train_kwargs, 
            val_dataset_kwargs=self.val_dataset_kwargs, 
            input_size=self.input_size,
            num_workers=self.cfg.data.num_workers,
            dist_world_size=self.trainer.world_size if self._trainer is not None else 1,
            dist_rank=self.trainer.global_rank if self._trainer is not None else 0,
        ) # WARNING: At this point device is CPU!!!

    def _loader_perf_kwargs(self):
        # pin_memory speeds host->GPU copies; persistent_workers/prefetch_factor only make
        # sense (and are only allowed) when there are worker processes.
        kwargs = dict(pin_memory=True)
        if self.cfg.data.num_workers > 0:
            # prefetch_factor=2 (the default): higher values × many workers × large batch hold
            # thousands of 640px tensors in RAM and OOM-kill the process on a 70 GB box.
            kwargs.update(persistent_workers=True, prefetch_factor=1)
        return kwargs

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
            **self._loader_perf_kwargs(),
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
            **self._loader_perf_kwargs(),
        )
        return dataloader


class LightlyModelMomentum(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        deactivate_requires_grad(self.backbone_momentum)

        self.projection_head_momentum = None

    def forward_momentum(self, x):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx):
        base_momentum = self.cfg.optimizer.get("ema_momentum", 0.996)
        momentum = cosine_schedule(self.current_epoch, self.cfg.trainer.max_epochs, base_momentum, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        if self.projection_head_momentum is not None:
            update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        return super().training_step(batch, batch_idx)


def main_pretrain(cfg: DictConfig, lightly_model: LightlyModel):
    print("Running on:", os.environ.get("HOSTNAME", "docker"), flush=True)
    os.system("nvidia-smi")
    print(torch.cuda.device_count(), "GPUs available", flush=True)

    # Use TF32 matmuls on Tensor Cores (Ampere/Ada) for faster fp32 ops.
    torch.set_float32_matmul_precision("high")
    # All pretrain configs use a fixed input size, so let cudnn benchmark conv algorithms
    # once per shape and reuse the fastest one.
    torch.backends.cudnn.benchmark = True

    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    OmegaConf.set_struct(cfg, False)

    cfg.artifacts_root += "_" + cfg.data.dataset_name

    flat_config = utils.flatten_dict(cfg)
    cfg.name = cfg.name.format(**flat_config)

    pl.seed_everything(cfg.seed)

    model = lightly_model(cfg)

    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            name=cfg.name, project="I-JEPA-CNN", save_dir="artifacts",
            group=cfg.get("wandb_group", None),
        )
        # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    root_dir = os.path.abspath(os.path.join(cfg.artifacts_root, cfg.name))
    version = utils.get_next_version(root_dir)
    ckpt_dir = os.path.join(root_dir, f"version_{version}")

    # Always emit a local CSV of the logged metrics (independent of wandb) so we have a robust
    # file to view feature_std. Writes to <ckpt_dir>/csv/metrics.csv.
    csv_logger = pl.loggers.CSVLogger(save_dir=ckpt_dir, name="csv")
    time.sleep(3) # To allow for other ranks to get the version number right
    if _get_rank() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    print("Checkpoint dir:", ckpt_dir, flush=True)
    
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,  # False to reduce disk load from constant checkpointing
        save_on_train_epoch_end=True,
        # save_top_k=1, # Doesn't work with DDP 
        # monitor="val_metrics/lin_top1",
        # mode="max",
    )
    callbacks = [checkpoint]

    # Note:
    # - DDP find_unused_parameters=False set because: https://pytorch-lightning.readthedocs.io/en/1.8.6/advanced/model_parallel.html?highlight=find_unused_parameter
    # - DDPStrategy vs DDPSpawnStrategy: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel-spawn
    # DDP only makes sense with >1 process. Under SLURM, SLURM_NTASKS gives the process count.
    # Off SLURM (e.g. a single-GPU Brev/cloud box) SLURM_NTASKS is unset: fall back to the
    # visible/available GPU count so a 1-GPU run uses the plain single-device strategy instead
    # of wrapping the model in DDP (which only adds overhead and a rendezvous on one GPU).
    world_size = os.environ.get("SLURM_NTASKS")
    if world_size is not None:
        world_size = int(world_size)
    else:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print("World size:", world_size, flush=True)
    if world_size <= 1:
        strategy = "auto"  # single device: no DDP wrapper
    else:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)

    loggers = [csv_logger]
    if cfg.wandb:
        loggers.insert(0, wandb_logger)
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        strategy=strategy,
        num_nodes=os.environ.get("SLURM_NNODES") or 1, # if SLURM_NNODES is not set, we assume 1 node
        **cfg.trainer,
    )
    print("CSV metrics ->", os.path.join(csv_logger.log_dir, "metrics.csv"), flush=True)
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}", flush=True)
    trainer.fit(model=model, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main_pretrain(LightlyModel)
