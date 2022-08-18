import logging
import time

import torch
import torch_geometric
from torch_geometric.graphgym.checkpoint import (clean_ckpt, load_ckpt, save_ckpt)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import (is_ckpt_epoch, is_eval_epoch, is_train_eval_epoch)
from torch_geometric.graphgym.loader import set_dataset_attr_eig
from torch_geometric.nn.glob.glob import global_add_pool, global_mean_pool, global_max_pool


total_feat_out = []
total_feat_in = []

def hook_fn_forward(module, input, output):
    if type(output) is tuple:
        output_clone = output[-1].clone()
    else:
        output_clone = output.clone()
    if type(input) is tuple:
        input_clone = input[-1].clone()
    else:
        input_clone = input.clone()
    total_feat_out.append(output_clone.detach())
    total_feat_in.append(input_clone.detach())



def proxy_pooling2(x):
    y = x.unsqueeze(0)
    layer = torch.nn.AvgPool2d((x.size(0), 1), stride=1) # transfer[27,300]->[1,300]
    # layer = torch.nn.MaxPool2d((1, x.shape[1]), stride=1) # transfer[27,300]->[27,1]
    y = layer(y)
    y = y.squeeze(0)
    y = y.t()
    return y


def compute_loss_proxy2(batch):
    """compute the loss, proxy_score and ground truth of proxy task (aka graph Laplacian eigenvector)"""
    # num_graphs = batch.num_graphs
    # compute first dataset in order to define loss function and score
    # data = batch[0]
    # net_feat = proxy_pooling2(data.x)
    # num_graphs = batch.num_graphs
    # for i in range(1, num_graphs):
    #     data = batch[i]
    #     data_feat = proxy_pooling2(data.x)
    #     net_feat = torch.cat([net_feat, data_feat], dim=0)
    x = batch.x
    net_feat = global_mean_pool(x, batch.batch)
    net_feat = net_feat.reshape([net_feat.numel(), 1])
    proxy_vec = batch.eig_vec
    proxy_true = proxy_vec
    loss, proxy_score = compute_loss(net_feat, proxy_vec)
    return loss, proxy_score, proxy_true


def attach_randomvec2(loaders):
    indices0 = loaders[0].dataset._indices
    indices1 = loaders[1].dataset._indices
    indices2 = loaders[2].dataset._indices
    full_length = indices0.__len__() + indices1.__len__() + indices2.__len__()
    for i in range(0, full_length):
        if i in indices0:
            loader = loaders[0]
            indices = indices0
        elif i in indices1:
            loader = loaders[1]
            indices = indices1
        else:
            loader = loaders[2]
            indices = indices2
        dataset = loader.dataset
        j = indices.index(i)
        data = dataset[j]
        size = cfg.gnn.dim_inner
        rand_vec = 2 * torch.rand([size, 1]) - 1  # In order to make the random vector distribute in range [-1, 1]
        rand_vec = rand_vec / torch.norm(rand_vec)
        if i == 0:
            # evec_all = compute_laplacian(data)
            rand_all = 2 * torch.rand([size, 1]) - 1
            rand_all = rand_all / torch.norm(rand_all)
        else:
            rand_all = torch.cat([rand_all, rand_vec], dim=0)
    rand_all = rand_all.to(torch.device('cpu'))
    length = len(dataset._data_list)
    slice = torch.linspace(0, length * cfg.gnn.dim_inner, steps=length+1)
    for i in range(0, loaders.__len__()):
        loader = loaders[i]
        set_dataset_attr_eig(loader.dataset, 'eig_vec', rand_all, slice)


def proxy_epoch2(logger, loader, model, optimizer, scheduler):
    global total_feat_in
    global total_feat_out
    model.train()
    time_start = time.time()
    for batch in loader:
        batch.split = 'train'
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred_train, true_train = model(batch)
        batch_hid = total_feat_in[-1]  # extract hidden layer feature
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, proxy_score, proxy_true = compute_loss_proxy2(batch_hid) # TODO (wby) here we redefined the compute_loss function
        # loss, pred_score = compute_loss(pred_train, true_train)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=proxy_true.detach().cpu(),
                            pred=proxy_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch2(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    """Register hook function is not needed here, because we have registered it on train_epoch function, and module will be
    hooked twice if we registered again."""
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred_eval, true_eval = model(batch)
        batch_hid = total_feat_in[-1][0]  # extract hidden layer feature
        del total_feat_in[-1]
        del total_feat_out[-1]  # delete item in order to save gpu memory
        loss, proxy_score, proxy_true = compute_loss_proxy2(batch_hid)
        # loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=proxy_true.detach().cpu(),
                            pred=proxy_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def proxy2(loggers, loaders, model, optimizer, scheduler):
    """
    The core proxy training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    # ==================================self-define hook function to extract hidden layer feature======================
    # for name0, module0 in model.named_children(): # TODO(wby) We will need this part of code later
    #     if name0=='mp':  # TODO(wby) Here we break mp layer into basic GNN layers.
    #         for name1, module1 in module0.named_children():
    #             module1.register_forward_hook(hook_fn_forward)
    #     else:
    #         module0.register_forward_hook(hook_fn_forward)
    for name0, module0 in model.named_children():
        if name0=='post_mp':
            module0.register_forward_hook(hook_fn_forward)
    # ================================================== self-define ends ============================================
    # ==================================compute dataset laplacian's eigenvectors=====================================
    # attach_eigenvec(loaders)
    # ================================================compute laplacian ends=========================================
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        proxy_epoch2(loggers[0], loaders[0], model, optimizer, scheduler)
        '''Determines if the model should be evaluated at the training epoch. The difference between is_train_eval_epoch
        and is_eval_epoch is that the user can self define whether logger should or not record the train-process-data
        (aka cfg.train.skip_train_eval), if cfg.train.skip_train_eval is True, loggers would record training-data
        '''
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)   # logger write for train datasets
        if is_eval_epoch(cur_epoch):            # Determines if the model should be evaluated at the current epoch.
            for i in range(1, num_splits):
                eval_epoch2(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))
