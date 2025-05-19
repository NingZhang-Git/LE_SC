import numpy as np
import scipy.sparse as sp
import networkx as nx

## Creat Di-SBM graph
# Parameters:
#     N : int
#         Total number of vertices.
#     k : int
#         Number of clusters.
#     community_sizes : list of int
#         Vector whose entries are community sizes.
#     p : float
#         Probability of edge within a cluster.
#     q : float
#         Probability of edge between clusters.
#     eta : float
#         Probability on edge from C2 to C1.
#     y : numpy.ndarray
#         True cluster labels {1,2,...,k}.

class DSBM():

    def __init__(self, N, k, community_sizes, p, q, eta_matrix):
        self.N = N
        self.k = k
        self.community_sizes = community_sizes
        self.p = p
        self.q = q
        self.eta_matrix = eta_matrix
        self.y = np.concatenate([np.full(ni, i + 1) for i, ni in enumerate(community_sizes)])

    def generate_graph(self):
        # Create probability matrix for SBM
        prob_matrix = np.full((self.k, self.k), self.q)
        np.fill_diagonal(prob_matrix, self.p)
        # Generate undirected SBM adjacency matrix
        G = nx.stochastic_block_model(self.community_sizes, prob_matrix)
        while not nx.is_connected(G):
            G = nx.stochastic_block_model(self.community_sizes, prob_matrix)

        A = nx.adjacency_matrix(G)
        # Create mask matrix (upper half) to determine edge direction
        di_mask = np.zeros((self.N, self.N))

        # Iterate through clusters to assign edge direction probabilities
        start_idx = 0
        for i in range(self.k):
            end_idx = start_idx + self.community_sizes[i]
            for j in range(self.k):
                block_start_idx_col = sum(self.community_sizes[:j])
                block_end_idx_col = block_start_idx_col+self.community_sizes[j]
                #print(i,j,start_idx, end_idx, block_start_idx_col, block_end_idx_col)
                if i==j:
                    # Within-cluster probability 0.5
                    di_mask[start_idx:end_idx, block_start_idx_col:block_end_idx_col] = np.random.binomial(1, 0.5, (self.community_sizes[i], self.community_sizes[j]))
                else:
                    # Between-cluster probability 1-eta_matrix[i, j]
                    di_mask[start_idx:end_idx, block_start_idx_col:block_end_idx_col] = np.random.binomial(1, 1-self.eta_matrix[i, j], (self.community_sizes[i], self.community_sizes[j]))
            start_idx = end_idx

        Ad_up = sp.triu(A*di_mask, k = 1)
        Ad_low = sp.tril(A - (Ad_up.T), k =-1)
        Ad =  Ad_up + Ad_low
        Ad_org = Ad.copy()

        assert((Ad+Ad.T - A).sum()==0.0)

        return Ad_org, self.y

