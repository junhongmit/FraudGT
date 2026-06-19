#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python -m fraudGT.main --cfg ${cfg_file}"
    out_dir="./results/${dataset}"
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    # Run once
    script="${main} --repeat 1 ${common_params}"
    echo $script
    eval $script
    echo $script
}

cfg_dir="configs"

#####Link Prediction Tasks###########################################################################
DATASET="AML-Small-HI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GatedGCN           "name_tag GatedGCN"
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False
# run_repeats ${DATASET} GINE+ports         "name_tag GINE+ports" #  wandb.use False
# run_repeats ${DATASET} GINE+RMP           "name_tag GINE+RMP" 
# run_repeats ${DATASET} GINE+Ego           "name_tag GINE+Ego" 
# run_repeats ${DATASET} GATE               "name_tag GATE"

# Proposed GT
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT" #+EgoID"
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT" #+EgoID"
run_repeats ${DATASET} SparseNodeGT+ports         "name_tag SparseNodeGT+ports"
# run_repeats ${DATASET} SparseNodeGT2+RMP          "name_tag SparseNodeGT2+RMP wandb.use False"
# run_repeats ${DATASET} SparseNodeGT+LP              "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+Dropout    "name_tag SparseNodeGT+Dropout"
# run_repeats ${DATASET} SparseNodeGT+Emb               "name_tag SparseNodeGT+Emb"
# run_repeats ${DATASET} SparseEdgeGT_Test               "name_tag SparseEdgeGT_Test"
# run_repeats ${DATASET} SparseEdgeGT+Metapath      "name_tag SparseEdgeGT+Metapath"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"


DATASET="AML-Small-LI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"

DATASET="AML-Medium-HI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"

DATASET="AML-Medium-LI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"

DATASET="AML-Large-HI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+Dropout       "name_tag SparseNodeGT+Dropout"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"

DATASET="AML-Large-LI"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+Dropout       "name_tag SparseNodeGT+Dropout"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"

DATASET="ETH"
# run_repeats ${DATASET} MLP                "name_tag MLP"
# run_repeats ${DATASET} MLP+Metapath                "name_tag MLP+Metapath"

# Homogeneous GNN Baselines
# run_repeats ${DATASET} GINE               "name_tag GINE" #  wandb.use False
# run_repeats ${DATASET} GINE+EU               "name_tag GINE+EU" #  wandb.use False
# run_repeats ${DATASET} GINE+RMP               "name_tag GINE+RMP" #  wandb.use False
# run_repeats ${DATASET} GINE+ports               "name_tag GINE+ports" #  wandb.use False
# run_repeats ${DATASET} GINE+Ego               "name_tag GINE+Ego" #  wandb.use False
# run_repeats ${DATASET} PNA               "name_tag PNA" #  wandb.use False
# run_repeats ${DATASET} PNA+EU               "name_tag PNA+EU" #  wandb.use False
# run_repeats ${DATASET} Multi-PNA               "name_tag Multi-PNA" #  wandb.use False
# run_repeats ${DATASET} Multi-PNA+EU               "name_tag Multi-PNA+EU" #  wandb.use False

# Proposed GT
# run_repeats ${DATASET} SparseNodeGT               "name_tag SparseNodeGT"
# run_repeats ${DATASET} SparseNodeGT+Node2Vec               "name_tag SparseNodeGT+Node2Vec"
# run_repeats ${DATASET} SparseNodeGT+ports               "name_tag SparseNodeGT+ports"
# run_repeats ${DATASET} SparseNodeGT+ports+Ego               "name_tag SparseNodeGT+ports+Ego"
# run_repeats ${DATASET} SparseNodeGT+LP               "name_tag SparseNodeGT+LP"
# run_repeats ${DATASET} SparseNodeGT+LP+ports               "name_tag SparseNodeGT+LP+ports"
# run_repeats ${DATASET} SparseNodeGT+RMP           "name_tag SparseNodeGT+RMP"
# run_repeats ${DATASET} Multi-SparseNodeGT               "name_tag Multi-SparseNodeGT"
# run_repeats ${DATASET} SparseEdgeGT               "name_tag SparseEdgeGT"
# run_repeats ${DATASET} GINE+SparseEdgeGT          "name_tag GINE+SparseEdgeGT"