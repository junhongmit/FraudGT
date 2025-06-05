import copy
import logging
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import mask_to_index, index_to_mask
from fraudGT.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import create_loader, get_loader
from fraudGT.graphgym.loss import compute_loss
from fraudGT.graphgym.register import register_train
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.optimizer import create_optimizer, create_scheduler
from fraudGT.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from fraudGT.graphgym.utils.comp_budget import params_count

from fraudGT.utils import cfg_to_dict, flatten_dict, make_wandb_name
from fraudGT.utils import (new_optimizer_config, new_scheduler_config)
from fraudGT.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats

def check_grad(model):
    for name, param in model.named_parameters():
        if 'attn_bias' in name:
            print(param.grad)
        # if param.requires_grad:
        #     if param.grad is None:
        #         print(f'{name} has no gradient')
        #     elif torch.isnan(param.grad).any():
        #         print(f'{name} has NaN gradients')
        #     elif torch.isinf(param.grad).any():
        #         print(f'{name} has Inf gradients')

# def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
#     model.train()
#     optimizer.zero_grad()
#     time_start = time.time()
#     for iter, batch in enumerate(tqdm(loader, disable=not cfg.train.tqdm)):
#         batch.split = 'train'
#         batch.to(torch.device(cfg.device))
#         pred, true = model(batch)

#         loss, pred_score = compute_loss(pred, true)
#         _true = true.detach().to('cpu', non_blocking=True)
#         _pred = pred_score.detach().to('cpu', non_blocking=True)

#         loss.backward()
#         # check_grad(model)
#         # Parameters update after accumulating gradients for given num. batches.
#         if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
#             if cfg.optim.clip_grad_norm:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                cfg.optim.clip_grad_norm_value)
#             optimizer.step()
#             optimizer.zero_grad()
#         cfg.params = params_count(model)
#         logger.update_stats(true=_true,
#                             pred=_pred,
#                             loss=loss.detach().cpu().item(),
#                             lr=scheduler.get_last_lr()[0],
#                             time_used=time.time() - time_start,
#                             params=cfg.params,
#                             dataset_name=cfg.dataset.name)
#         time_start = time.time()

def train_epoch(cur_epoch, logger, loader, model, optimizer, scheduler, batch_accumulation):
    pbar = tqdm(total=len(loader), disable=not cfg.train.tqdm)
    pbar.set_description(f'Train epoch')

    model.train()

    runtime_stats_cuda.start_epoch()

    runtime_stats_cuda.start_region("total")
    runtime_stats_cuda.start_region(
        "sampling", runtime_stats_cuda.get_last_event())
    iterator = iter(loader)
    runtime_stats_cuda.end_region("sampling")
    runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())

    if cfg.model.type == 'LPModel': # Handle label propagation specially
        # We don't need to train label propagation
        time_start = time.time()
        batch = next(iterator, None)
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=0,
                            dataset_name=cfg.dataset.name)
        pbar.update(1)
        return

    optimizer.zero_grad()
    it = 0
    time_start = time.time()
    # with torch.autograd.set_detect_anomaly(True):
    while True:
        try:
            torch.cuda.empty_cache() 
            runtime_stats_cuda.start_region(
                "total", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region(
                "sampling", runtime_stats_cuda.get_last_event())
            # print('Crashed?')
            batch = next(iterator, None)
            # print('Crashed?')
            it += 1
            if batch is None:
                runtime_stats_cuda.end_region("sampling")
                runtime_stats_cuda.end_region(
                    "total", runtime_stats_cuda.get_last_event())
                break
            runtime_stats_cuda.end_region("sampling")

            runtime_stats_cuda.start_region("data_transfer", runtime_stats_cuda.get_last_event())
            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = 'train'
                batch.to(torch.device(cfg.device))
            else: # NAGphormer, HINo
                batch = [x.to(torch.device(cfg.device)) for x in batch]
            runtime_stats_cuda.end_region("data_transfer")

            runtime_stats_cuda.start_region("train", runtime_stats_cuda.get_last_event())
            runtime_stats_cuda.start_region("forward", runtime_stats_cuda.get_last_event())
            pred, true = model(batch)
            runtime_stats_cuda.end_region("forward")
            runtime_stats_cuda.start_region("loss", runtime_stats_cuda.get_last_event())
            if cfg.model.loss_fun == 'curriculum_learning_loss':
                loss, pred_score = compute_loss(pred, true, cur_epoch)
            else:
                loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            _ids = batch[cfg.dataset.task_entity].input_id.detach().to('cpu', non_blocking=True)
            runtime_stats_cuda.end_region("loss")

            runtime_stats_cuda.start_region("backward", runtime_stats_cuda.get_last_event())
            loss.backward()
            runtime_stats_cuda.end_region("backward")
            # print(loss.detach().cpu().item())
            # check_grad(model)
            # Parameters update after accumulating gradients for given num. batches.
            if ((it + 1) % batch_accumulation == 0) or (it + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg.optim.clip_grad_norm_value)
                optimizer.step()
                optimizer.zero_grad()
            runtime_stats_cuda.end_region("train")
            runtime_stats_cuda.end_region("total", runtime_stats_cuda.get_last_event())
            cfg.params = params_count(model)
            logger.update_stats(ids=_ids,
                                true=_true,
                                pred=_pred,
                                loss=loss.detach().cpu().item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise
    
    runtime_stats_cuda.end_epoch()
    runtime_stats_cuda.report_stats(
        {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train', \
         'attention': 'Attention', 'gt-layer': 'GT-Layer', 'forward': 'Forward', 'loss': 'Loss', 'backward': 'Backward'})


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', log_predictions: bool = False, log_embedding: bool = False):
    model.eval()
    pbar = tqdm(total=len(loader), disable=not cfg.val.tqdm)
    iterator = iter(loader)

    time_start = time.time()
    it = 0
    while True:
        try:
            batch = next(iterator, None)
            it += 1
            if batch is None:
                break
            if isinstance(batch, Data) or isinstance(batch, HeteroData):
                batch.split = split
                batch.to(torch.device(cfg.device))
            else: # NAGphormer
                batch = [x.to(torch.device(cfg.device)) for x in batch]
            if cfg.gnn.head == 'inductive_edge':
                pred, true, extra_stats = model(batch)
            else:
                if (log_embedding):
                    pred, true, embs = model(batch, return_embedding=True)
                else:
                    pred, true = model(batch)
                extra_stats = {}

            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
            _ids = batch[cfg.dataset.task_entity].input_id.detach().to('cpu', non_blocking=True)
            
            logger.update_stats(ids=_ids,
                                true=_true,
                                pred=_pred,
                                embs=embs if log_embedding else None,
                                loss=loss.detach().cpu().item(),
                                lr=0, time_used=time.time() - time_start,
                                params=cfg.params,
                                dataset_name=cfg.dataset.name,
                                **extra_stats)
            pbar.update(1)
            time_start = time.time()
        except RuntimeError as e:
            if "cannot sample n_sample <= 0 samples" in str(e):
                print(f"Skipping batch due to error: {e}")
                continue
            else:
                # If it's a different error, re-raise it
                raise
    if log_predictions:
        logger.export_prediction(f"{cfg.pretrained.prediction_log}/predictions_{split}.pt")


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name, dir='/nobackup/users/junhong/Logs/')
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        # enable_runtime_stats()
        train_epoch(cur_epoch, loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        # disable_runtime_stats()
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch, start_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch, start_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i], log_predictions=True, log_embedding=True)
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from fraudGT.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.device))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
