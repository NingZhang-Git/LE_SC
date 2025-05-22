# Spectral Methods for Directed Graph Clustering

This repository contains experiments and analysis for clustering **directed graphs** using three datasets:

- **US Migration** — internal migration patterns between US states.
- **Connectome** — neuronal connectivity data.
- **DSBM** — Directed Stochastic Block Model, a synthetic benchmark for evaluating clustering algorithms.

```
├── Connectome/                    # Neuron connectivity clustering
│   ├── adjacency.npz             # Adjacency matrix of neural network
│   ├── labels.npy                # Ground truth cluster labels
│   └── Neuron cluster.ipynb      # Notebook for clustering neuron data
│
├── DSBM/                         # DSBM clustering experiments
│   ├── Cluster DSBM_k2.ipynb     # Clustering DSBM with k=2
│
├── US migration/                 # US county migration clustering
│   ├── Coord.csv                 # Coordinates of US counties
│   ├── MIG.csv                   # Migration data between counties
│   ├── US_county_cluster.ipynb   # Notebook for clustering US counties
│   └── USA.txt                   # Supplementary US data
│
├── utils/                        # Utility scripts
│   ├── ClusterAlgorithms.py      # Clustering algorithm implementations
│   ├── RandomGraph.py            # Random graph generators
│   └── USA_Plot.py               # Plotting tools for US maps

```
## Key Algorithm

The file `utils/ClusterAlgorithms.py` includes multiple baseline methods for clustering directed graphs. Our proposed method, LE-SC (Likelihood Estimation Spectral Clustering), is an iterative spectral clustering algorithm that refines clusters through a likelihood-based update process to produce final community labels.

## Requirements

- Python 3.7+
- Common scientific libraries: `numpy`, `scipy`, `networkx`, `matplotlib`

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd <repo-folder>

# (Optional) create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
