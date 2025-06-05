from fraudGT.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    unifiedGT.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Training (and validation) pipeline mode
    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = False

    # Enable tqdm progress bar during training/validation/testing
    cfg.train.tqdm = False
    cfg.val.tqdm = False

    # In training, when the graph is very large, you may not want to iterate through
    # all the batches available. You can sample a subset of available batches as
    # the HGT does in their code.
    cfg.train.iter_per_epoch = 0

    # In evaluation, you may want to reduce the evaluation time like the above rationale
    # for training. However, to reduce the periodic variance due to cycle through the val/test
    # set, we fixed the evaluation sampling to the same set of batches. So the performance
    # would be evaluated under the same set of data.
    cfg.val.iter_per_epoch = 0

    cfg.train.shuffle = True
    cfg.val.shuffle= True

    # Sampling parameters
    cfg.train.persistent_workers = False
    cfg.train.pin_memory = False

    # NeighborSampler / HGTSampler: number of sampled nodes per layer for each node type
    cfg.train.neighbor_sizes_dict = ""

    # RandomNodeLoader: number of partitions
    cfg.train.num_parts = 10

    # AddEgoID option
    cfg.train.add_ego_id = False

    # APPNP hyperparameter
    cfg.gnn.K = 10
    cfg.gnn.alpha = 0.1

    # NAGphormer hop2seq hyperparameter
    cfg.gnn.hops = 7

    # SHGN hyperparameter
    cfg.gnn.residual = True

    # LSGNN hyperparameter
    cfg.gnn.A_embed = True

    cfg.gnn.batch_norm = False
    cfg.gnn.layer_norm = False

    cfg.gnn.input_dropout = 0.0

    cfg.gnn.attn_dropout = 0.0

    cfg.gnn.edge_updates = False
    cfg.gnn.use_linear = False
    cfg.gnn.output_l2_norm = False
    cfg.gnn.jumping_knowledge = False

    cfg.model.loss_fun_weight = []