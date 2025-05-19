# Directed Graph Clustering Experiments

This repository contains experiments and analysis for clustering **directed graphs** using three datasets:

- **US Migration** — internal migration patterns between US states.
- **Connectome** — neuronal connectivity data.
- **DSBM** — Directed Stochastic Block Model, a synthetic benchmark for evaluating clustering algorithms.

## Structure

├── DSBM/ # Directed Stochastic Block Model experiments
├── US migration/ # US migration graph analysis
├── Connectome/ # Neuron connectome analysis
├── utils/ # Utility scripts (graph generation, clustering, visualization)
│ ├── ClusterAlgorithms.py
│ ├── RandomGraph.py
│ ├── USA_Plot.py
│ └── pycache/
├── README.md # This file



## Contents

### 📁 `utils/`

Contains reusable tools for:

- Generating synthetic directed graphs (`RandomGraph.py`)
- Implementing and applying clustering algorithms (`ClusterAlgorithms.py`)
- Visualizing US migration graph data (`USA_Plot.py`)

These utility modules are used across all three datasets for consistency.

### 📁 `DSBM/`

Includes experiments with the Directed Stochastic Block Model, useful for validating the performance of clustering algorithms under controlled conditions.

### 📁 `Connectome/`

Processes and clusters directed graphs derived from real-world neuronal connectome data.

### 📁 `US migration/`

Analyzes directed migration flows between US states and evaluates the structure using graph clustering methods.

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
