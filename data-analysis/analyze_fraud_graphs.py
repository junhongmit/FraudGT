"""
Fraud review graph analysis: Amazon & YelpChi.

Downloads both datasets via DGL, converts to NetworkX for metric computation,
and generates plots saved to data-analysis/figures/.

Run from the repo root:
    python data-analysis/analyze_fraud_graphs.py
"""

import os
import warnings
import numpy as np
import torch
import dgl
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter

warnings.filterwarnings('ignore')

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Dataset loading ────────────────────────────────────────────────────────────

DATASET_CFGS = {
    'Amazon': {
        'dgl_cls':   dgl.data.FraudAmazonDataset,
        'node_type': 'user',
        'edge_types': ['net_upu', 'net_usu', 'net_uvu'],
        'feat_key':  'feature',
    },
    'YelpChi': {
        'dgl_cls':   dgl.data.FraudYelpDataset,
        'node_type': 'review',
        'edge_types': ['net_rsr', 'net_rtr', 'net_rur'],
        'feat_key':  'feature',
    },
}


def load_dataset(name):
    cfg = DATASET_CFGS[name]
    raw_dir = os.path.join('data-analysis', 'raw', name)
    os.makedirs(raw_dir, exist_ok=True)
    dataset = cfg['dgl_cls'](raw_dir=raw_dir, verbose=False)
    g = dataset[0]
    return g, cfg


# ── Preprocessing analysis ─────────────────────────────────────────────────────

def analyze_preprocessing(name, g, cfg):
    print(f'\n{"="*60}')
    print(f'  PREPROCESSING — {name}')
    print(f'{"="*60}')

    feat_key = cfg['feat_key']
    x = g.ndata[feat_key].numpy()
    y = g.ndata['label'].numpy()

    print(f'\n[Node features]')
    print(f'  Shape          : {x.shape}')
    print(f'  Dtype          : {x.dtype}')
    print(f'  Missing values : {np.isnan(x).sum()} NaN, {np.isinf(x).sum()} Inf')
    print(f'  Duplicate rows : {x.shape[0] - len(np.unique(x, axis=0))}')
    print(f'  Feature range  : [{x.min():.4f}, {x.max():.4f}]')
    print(f'  Feature mean   : {x.mean():.4f}  std: {x.std():.4f}')

    # Per-feature stats
    feat_means = x.mean(axis=0)
    feat_stds  = x.std(axis=0)
    zero_var   = (feat_stds == 0).sum()
    print(f'  Zero-variance features: {zero_var}')

    # Label stats
    counts = Counter(y)
    total  = len(y)
    print(f'\n[Labels]')
    print(f'  Total nodes : {total:,}')
    for cls, cnt in sorted(counts.items()):
        tag = 'fraudulent/spam' if cls == 1 else 'legitimate'
        print(f'  Class {cls} ({tag}): {cnt:,} ({cnt/total*100:.2f}%)')
    print(f'  Imbalance ratio (legit:fraud): {counts[0]/counts[1]:.1f}:1')

    # After z-normalization (what FraudGT applies)
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    print(f'\n[After z-normalization]')
    print(f'  Mean: {x_norm.mean():.6f}  Std: {x_norm.std():.6f}')

    return x, y, x_norm


# ── Graph construction analysis ────────────────────────────────────────────────

def analyze_graph_construction(name, g, cfg):
    print(f'\n{"="*60}')
    print(f'  GRAPH CONSTRUCTION — {name}')
    print(f'{"="*60}')

    node_type  = cfg['node_type']
    edge_types = cfg['edge_types']
    num_nodes  = g.num_nodes()

    print(f'\n[Nodes]')
    print(f'  Type            : {node_type}')
    print(f'  Count           : {num_nodes:,}')
    print(f'  Feature dim     : {g.ndata[cfg["feat_key"]].shape[1]}')

    print(f'\n[Edges — per relation type]')
    total_edges = 0
    edge_info = {}
    for et in edge_types:
        n_edges = g.num_edges(etype=et)
        total_edges += n_edges
        src, dst = g.edges(etype=et)
        density = n_edges / (num_nodes * (num_nodes - 1))
        print(f'  {et}:')
        print(f'    Directed edges : {n_edges:,}')
        print(f'    Density        : {density:.6f} ({density*100:.4f}%)')
        edge_info[et] = {'n_edges': n_edges, 'src': src, 'dst': dst}

    print(f'\n  Total directed edges (all types): {total_edges:,}')
    print(f'  Relation types: {len(edge_types)} (heterogeneous homophily graph)')

    # Edge attributes note
    has_efeats = any(len(g.edata) > 0 for _ in edge_types)
    print(f'\n[Edge features]')
    print(f'  Present: {has_efeats}')
    if not has_efeats:
        print(f'  → Edges carry no features; model uses structural signal only.')

    print(f'\n[PyG HeteroData structure]')
    print(f'  data["{node_type}"].x          : float [{num_nodes}, {g.ndata[cfg["feat_key"]].shape[1]}]')
    print(f'  data["{node_type}"].y          : long  [{num_nodes}]')
    for et in edge_types:
        n = edge_info[et]['n_edges']
        print(f'  data["{node_type}", "{et}", "{node_type}"].edge_index : long [2, {n}]')

    return edge_info


# ── Structural analysis ────────────────────────────────────────────────────────

def analyze_structure(name, g, cfg, edge_info):
    print(f'\n{"="*60}')
    print(f'  STRUCTURAL ANALYSIS — {name}')
    print(f'{"="*60}')

    edge_types = cfg['edge_types']
    num_nodes  = g.num_nodes()
    y = g.ndata['label'].numpy()

    for et in edge_types:
        src = edge_info[et]['src'].numpy()
        dst = edge_info[et]['dst'].numpy()
        n_edges = edge_info[et]['n_edges']

        # Degree (treating as undirected: count in+out but only unique pairs)
        # DGL stores undirected as two directed edges, so in-degree == out-degree
        in_deg = np.bincount(dst, minlength=num_nodes)

        print(f'\n  [{et}] — node degree distribution')
        print(f'    Min degree    : {in_deg.min()}')
        print(f'    Max degree    : {in_deg.max()}')
        print(f'    Mean degree   : {in_deg.mean():.2f}')
        print(f'    Median degree : {np.median(in_deg):.2f}')
        print(f'    Std degree    : {in_deg.std():.2f}')
        print(f'    Isolated nodes: {(in_deg == 0).sum():,} ({(in_deg==0).mean()*100:.1f}%)')

        # Degree by class
        fraud_deg = in_deg[y == 1]
        legit_deg = in_deg[y == 0]
        print(f'    Mean degree (fraud) : {fraud_deg.mean():.2f}')
        print(f'    Mean degree (legit) : {legit_deg.mean():.2f}')

    # Graph-level metrics (use lightest relation type for NX analysis)
    lightest_et = min(edge_types, key=lambda e: edge_info[e]['n_edges'])
    src = edge_info[lightest_et]['src'].numpy()
    dst = edge_info[lightest_et]['dst'].numpy()

    print(f'\n  [Graph-level metrics on "{lightest_et}" (lightest relation)]')
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(src, dst))

    n_components = nx.number_connected_components(G)
    largest_cc   = max(nx.connected_components(G), key=len)
    print(f'    Connected components : {n_components}')
    print(f'    Largest CC size      : {len(largest_cc):,} nodes ({len(largest_cc)/num_nodes*100:.1f}%)')

    # Clustering coefficient on subgraph sample (full graph is too slow)
    sample_size = min(2000, len(largest_cc))
    sample_nodes = np.random.choice(list(largest_cc), size=sample_size, replace=False)
    subG = G.subgraph(sample_nodes)
    cc = nx.average_clustering(subG)
    print(f'    Avg clustering coeff (sample n={sample_size}): {cc:.4f}')

    return G, lightest_et


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_class_distribution(name, y):
    counts = Counter(y)
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    values = [counts[0], counts[1]]
    colors = ['#4CAF50', '#F44336']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(y)*0.005,
                f'{val:,}\n({val/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_title(f'{name} — Class Distribution', fontsize=13)
    ax.set_ylabel('Number of nodes')
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{name}_class_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_degree_distribution(name, g, cfg, edge_info):
    edge_types = cfg['edge_types']
    num_nodes  = g.num_nodes()
    fig, axes = plt.subplots(1, len(edge_types), figsize=(5*len(edge_types), 4))
    if len(edge_types) == 1:
        axes = [axes]

    for ax, et in zip(axes, edge_types):
        dst = edge_info[et]['dst'].numpy()
        in_deg = np.bincount(dst, minlength=num_nodes)
        # Cap at 99th percentile for readability
        cap = np.percentile(in_deg[in_deg > 0], 99)
        deg_capped = in_deg[in_deg <= cap]
        ax.hist(deg_capped, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
        ax.set_title(f'{et}', fontsize=11)
        ax.set_xlabel('Degree (in, capped at p99)')
        ax.set_ylabel('Count')
        ax.set_yscale('log')

    fig.suptitle(f'{name} — Degree Distributions (log scale)', fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{name}_degree_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_subgraph(name, G, lightest_et, y, max_nodes=200):
    """Visualize a small ego-network subgraph coloured by class label."""
    # Pick a high-degree seed node to get a connected subgraph
    seed = max(G.degree, key=lambda x: x[1])[0]
    nodes = list(nx.ego_graph(G, seed, radius=2).nodes)[:max_nodes]
    subG  = G.subgraph(nodes)
    colors = ['#F44336' if y[n] == 1 else '#4CAF50' for n in subG.nodes]

    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(subG, seed=42, k=0.4)
    nx.draw_networkx_nodes(subG, pos, node_color=colors, node_size=40, ax=ax)
    nx.draw_networkx_edges(subG, pos, alpha=0.2, ax=ax)
    ax.set_title(f'{name} — Subgraph (relation: {lightest_et}, n={len(nodes)})\n'
                 f'Red = fraud/spam, Green = legitimate', fontsize=11)
    ax.axis('off')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#F44336', label='Fraud/Spam'),
                       Patch(facecolor='#4CAF50', label='Legitimate')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{name}_subgraph_{lightest_et}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_feature_heatmap(name, x_norm, y, n_features=25):
    """Mean feature value per class (heatmap)."""
    feat_fraud = x_norm[y == 1, :n_features].mean(axis=0)
    feat_legit = x_norm[y == 0, :n_features].mean(axis=0)
    data = np.vstack([feat_legit, feat_fraud])

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Legitimate', 'Fraud/Spam'])
    ax.set_xlabel('Feature index')
    ax.set_title(f'{name} — Mean feature value per class (z-normalised, first {n_features} features)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'{name}_feature_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for name in ['Amazon', 'YelpChi']:
        print(f'\n{"#"*60}')
        print(f'#  {name}')
        print(f'{"#"*60}')

        g, cfg = load_dataset(name)
        x, y, x_norm = analyze_preprocessing(name, g, cfg)
        edge_info     = analyze_graph_construction(name, g, cfg)
        G, lightest   = analyze_structure(name, g, cfg, edge_info)

        print(f'\n[Generating figures]')
        plot_class_distribution(name, y)
        plot_degree_distribution(name, g, cfg, edge_info)
        plot_subgraph(name, G, lightest, y)
        plot_feature_heatmap(name, x_norm, y, n_features=min(25, x_norm.shape[1]))

    print('\nDone. All figures saved to data-analysis/figures/')


if __name__ == '__main__':
    main()
