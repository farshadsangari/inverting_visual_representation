"""
Utilities of Project
"""
import numpy as np
import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def get_args():
    """
    The argument that we have defined, will be used in training and evaluation(infrence) modes
    """
    parser = argparse.ArgumentParser(
        description="Arguemnt Parser of `Train` and `Evaluation` of our network"
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=64,
        type=int,
        help="Number of data in each batch",
    )

    parser.add_argument(
        "--lr", dest="lr", default=1e-3, type=float, help="Learning rate value"
    )

    parser.add_argument(
        "--momentum",
        dest="momentum",
        default=0.9,
        type=float,
        help="Momentum coefficient",
    )

    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay value",
    )

    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        default=30,
        type=int,
        help="Number of epochs",
    )

    parser.add_argument(
        "--gpu", dest="gpu", default=True, type=bool, help="wheather to use gpu or not"
    )

    parser.add_argument(
        "--ckpt-save-path",
        dest="ckpt_save_path",
        default="../ckpts",
        type=str,
        help="base path(folder) to save model ckpts",
    )

    parser.add_argument(
        "--ckpt-prefix",
        dest="ckpt_prefix",
        default="cktp_epoch_",
        type=str,
        help="prefix name of ckpt which you want to save",
    )

    parser.add_argument(
        "--ckpt-save-freq",
        dest="ckpt_save_freq",
        default=10,
        type=int,
        help="after how many epoch(s) save model",
    )

    parser.add_argument(
        "--ckpt-load-path",
        dest="ckpt_load_path",
        type=str,
        default=None,
        help="Checkpoints address for loading",
    )

    parser.add_argument(
        "--report-path",
        dest="report_path",
        type=str,
        default="../reports",
        help="Saving report directory",
    )

    parser.add_argument(
        "--all-data-pathes",
        dest="all_data_pathes",
        type=str,
        default="./data/tiny-imagenet-200/train/",
        help="Path of all images(x)",
    )

    parser.add_argument(
        "--regex-for-category",
        dest="regex_for_category",
        type=str,
        default="\/data\/tiny-imagenet-200\/train\/(.*)\/images\/.*",
        help="Regex of images(x) which will be used for category",
    )
    parser.add_argument(
        "--betas",
        dest="betas",
        nargs=2,
        type=float,
        default=(0.9, 0.999),
        help="Betas argument for adam optimizer",
    )
    parser.add_argument(
        "--gamma",
        dest="gamma",
        default=0.5,
        type=float,
        help="Gamma coefficient for learning rate schedular",
    )

    parser.add_argument(
        "--step-size",
        dest="step_size",
        default=64,
        type=int,
        help="Step size for learning rate schedular",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=device,
        help="cpu or gpu(x)",
    )

    options = parser.parse_args()

    return options


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save

    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group["params"], max_norm, norm_type)


def Interpolate(x):
    if len(x.shape) == 4:
        return F.interpolate(x, 227)
    if len(x.shape) == 3:
        return (F.interpolate(x[None, :, :, :], 227)).view(3, 227, 227)
