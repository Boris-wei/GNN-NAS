import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_agg_dir, set_run_dir)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.pre_train import pre_train
from torch_geometric.graphgym.utils.agg_runs import agg_runs, is_seed, json_to_dict_list, dict_to_json, is_split
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.proxy import attach_eigenvec, attach_randomvec
import json

def delete_pre_json(dir):
    stats_list = json_to_dict_list(dir)
    del stats_list[0:20]
    with open(dir, 'r+') as f:
        f.truncate()    # clear json file
    for i in range(0, stats_list.__len__()):
        dict_to_json(stats_list[i], dir)


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)  # 这里cfg由命令行进行指定，是一个.yaml文件
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    loaders = create_loader()  # list of loaders, they are divided from original dataset according to 'train' 'test' and 'val'
    attach_randomvec(loaders)
    for i in range(args.repeat):
        cfg.model.loss_fun = 'mse'
        cfg.dataset.task_type = 'regression'
        set_run_dir(cfg.out_dir, args.cfg_file)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed) # Sets the seed for generating random numbers in PyTorch, numpy and Python.实际上就是生成随机数，避免训练结果相同
        auto_select_device()
        # Set machine learning pipeline
        # loaders = create_loader() # list of loaders, they are divided from original dataset according to 'train' 'test' and 'val'
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters(), cfg.optim)
        scheduler = create_scheduler(optimizer, cfg.optim)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        # attach_randomvec(loaders)
        if cfg.train.mode == 'standard':
            pre_train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # delete pre-train results
    dir = set_agg_dir(cfg.out_dir, args.cfg_file)
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)
            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    delete_pre_json(fname_stats)



    # Aggregate results from different seeds
    agg_runs(set_agg_dir(cfg.out_dir, args.cfg_file), cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfmain_proxy_random.pyg_file, f'{args.cfg_file}_done')
