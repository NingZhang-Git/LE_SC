# Spectral Methods for Directed Graph Clustering

This repository contains experiments and analysis for clustering **directed graphs** using three datasets:

- **US Migration** — internal migration patterns between US states.
- **Connectome** — neuronal connectivity data.
- **DSBM** — Directed Stochastic Block Model, a synthetic benchmark for evaluating clustering algorithms.



## Contents

### 📁 `utils/`

Contains reusable tools for:

- Generating synthetic directed graphs (`RandomGraph.py`)
- Implementing and applying spectral clustering algorithms (`ClusterAlgorithms.py`)
- Visualizing US migration graph data (`USA_Plot.py`)

These utility modules are used across all three datasets for consistency.

### 📁 `DSBM/`

Clustering directed graphs generated from the Directed Stochastic Block Model.

### 📁 `Connectome/`

Clusters directed graphs representing the Larval Drosophila mushroom body connectome.

### 📁 `US migration/`

Analyzes directed migration flows between US counties from 1995 to 2000.

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
