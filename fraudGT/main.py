import datetime, os, logging
import torch

import fraudGT  # noqa, register custom modules
from fraudGT.agg_runs import agg_runs
from fraudGT.optimizer.extra_optimizers import ExtendedSchedulerConfig

from fraudGT.graphgym.cmd_args import parse_args
from fraudGT.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from fraudGT.graphgym.loader import create_loader
from fraudGT.graphgym.logger import setup_printing
from fraudGT.graphgym.optimizer import create_optimizer, \
    create_scheduler, OptimizerConfig
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.train import train
from fraudGT.graphgym.utils.comp_budget import params_count
from fraudGT.graphgym.utils.device import auto_select_device
from fraudGT.graphgym.register import train_dict
from torch_geometric import seed_everything

from fraudGT.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from fraudGT.logger import create_logger
from fraudGT.utils import (new_optimizer_config, new_scheduler_config, \
                             custom_set_out_dir, custom_set_run_dir)


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def run():
    global cfg
    def run_loop_settings():
        """Create main loop execution settings based on the current cfg.

        Configures the main execution loop to run in one of two modes:
        1. 'multi-seed' - Reproduces default behaviour of GraphGym when
            args.repeats controls how many times the experiment run is repeated.
            Each iteration is executed with a random seed set to an increment from
            the previous one, starting at initial cfg.seed.
        2. 'multi-split' - Executes the experiment run over multiple dataset splits,
            these can be multiple CV splits or multiple standard splits. The random
            seed is reset to the initial cfg.seed value for each run iteration.

        Returns:
            List of run IDs for each loop iteration
            List of rng seeds to loop over
            List of dataset split indices to loop over
        """
        if len(cfg.run_multiple_splits) == 0:
            # 'multi-seed' run mode
            num_iterations = args.repeat
            seeds = [cfg.seed + x for x in range(num_iterations)]
            split_indices = [cfg.dataset.split_index] * num_iterations
            run_ids = seeds
        else:
            # 'multi-split' run mode
            if args.repeat != 1:
                raise NotImplementedError("Running multiple repeats of multiple "
                                        "splits in one run is not supported.")
            num_iterations = len(cfg.run_multiple_splits)
            seeds = [cfg.seed] * num_iterations
            split_indices = cfg.run_multiple_splits
            run_ids = split_indices
        return run_ids, seeds, split_indices

    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        setup_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        if args.gpu == -1:
            auto_select_device(strategy='greedy')
        else:
            logging.info('Select GPU {}'.format(args.gpu))
            if cfg.device == 'auto':
                cfg.device = 'cuda:{}'.format(args.gpu)
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        # Set machine learning pipeline
        loaders, dataset = create_loader(train_shuffle=cfg.train.shuffle, val_shuffle=cfg.val.shuffle, returnDataset=True)
        loggers = create_logger()
        model = create_model(dataset=dataset)
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head, seed=cfg.seed
            )
        optimizer = create_optimizer(model.named_parameters(), #model.named_parameters(),
                                     new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            # if cfg.wandb.use:
            #     logging.warning("[W] WandB logging is not supported with the "
            #                     "default train.mode, set it to `custom`")
            # datamodule = GraphGymDataModule()
            # train(model, datamodule, logger=True)
            pass
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")

if __name__ == '__main__':
    run()