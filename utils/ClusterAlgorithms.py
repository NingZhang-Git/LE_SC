import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
import scipy.sparse as sp
import pdb
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.linalg import LinearOperator

from math import ceil
from math import pi
from math import comb
from itertools import combinations
from cmath import exp

class ClusterBy():
    def __init__(self,Ad,k):

      self.Ad = Ad
      self.k = k
      self.N = Ad.shape[0]

    def eff_sparse_eig(self, M_sp, eps, k):
      ones = np.ones(self.N)
      def matvec(x):
        return M_sp @ x + eps * np.sum(x) * ones

      M_eff = LinearOperator((self.N, self.N), matvec=matvec)
      vals, vecs = sp.linalg.eigsh(M_eff, k)
      return vals, vecs

    def HermCLuster(self):
      """
        Hermitian clustering algorithm.
        Parameters:
            Ad (numpy.ndarray): Graph adjacency matrix
            k (int): Number of clusters, in this study k=2
        Returns:
            numpy.ndarray: Cluster label vector
      """
        # Step 1. Compute Hermitian adjacency matrix
      H = 1j* (self.Ad - self.Ad.T)
        # Step 2. Compute the eigenvectors of H
      val, v = sp.linalg.eigsh(H, k=int(ceil(self.k/2)))
        # Step 3. k-means clustering on eigenvectors
      V_H = np.hstack((v.real, v.imag))  # shape: (N,2k)
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat = kmeans.fit_predict(V_H)
      return y_hat
    
    def HermRWCluster(self):
      """
        Hermitian random walk clustering algorithm.
        Parameters:
            Ad (numpy.ndarray): Graph adjacency matrix
            k (int): Number of clusters, in this study k=2
        Returns:
            numpy.ndarray: Cluster label vector
      """
      # Step 1. Compute the Hermitian RW matrix
      P = np.sum(self.Ad,axis = 1)
      O = np.sum(self.Ad,axis = 0)
      O = np.array(O).flatten()
      P = np.array(P).flatten()
      A_rw = np.diag(1.0 / (O+P)) @ self.Ad
      Hrw = 1j* (A_rw - A_rw.T)
      # Step 2. Compute the eigenvectors of Hrw
      val, v = sp.linalg.eigs(Hrw, k=int(ceil(self.k/2)))
      # Step 3. k-means clustering on eigenvectors
      V_Hrw = np.hstack((v.real, v.imag))  # shape: (N,2k)
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_HermRW = kmeans.fit_predict(V_Hrw)
      return y_HermRW
      

    def SpectralCluster(self):
      """
        Spectral clustering algorithm.
        Parameters:
            Ad (numpy.ndarray): Graph adjacency matrix
            k (int): Number of clusters, in this study k=2
        Returns:
            numpy.ndarray: Cluster label vector
      """
      # Step 1. Compute the normalized adjacency matrix
      A = self.Ad + self.Ad.T
      # Step 2. Compute the eigenvecgtors of A
      val, v = sp.linalg.eigsh(A, k=self.k)
      # Step 3. k-means clustering on eigenvectors
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat = kmeans.fit_predict(v.real)
      return y_hat

    def DISIM(self, reg =1):
      """
        DI_SIM Cluster algorithm.
        Parameters:
            Ad (numpy.ndarray): Graph adjacency matrix
            k (int): Number of clusters, in this study k=2
            reg (float): Regularization parameter (0 = no regularization)
        Returns:
            numpy.ndarray: Cluster label vector
      """
      Ad = self.Ad
      O = np.sum(Ad, axis=1)  # Sum over rows
      P = np.sum(Ad, axis=0)  # Sum over columns
      O = np.array(O).flatten()
      P = np.array(P).flatten()
        # Step 1. Compute the normalized (regularized) adjacency matrix
      if reg ==0:
          O_norm = O.copy()
          P_norm = P.copy()
          O_norm[O_norm ==0] = 1
          P_norm[P_norm ==0] = 1
          A_norm = np.diag(1.0 / np.sqrt(O_norm)) @ Ad @ np.diag(1.0 / np.sqrt(P_norm))
      else:
          t = np.sum(O) / (Ad.shape[0]) #average out degree
          O_norm = O + t
          P_norm = P + t
          A_norm = np.diag(1.0 / np.sqrt(O_norm)) @ Ad @ np.diag(1.0 / np.sqrt(P_norm))
        # Step 2. Compute leading eigenvector of AA' and A'A
      val_L, v_disim_L = sp.linalg.eigsh(A_norm @ A_norm.T, self.k, maxiter=10000)
      val_R, v_disim_R  = sp.linalg.eigsh(A_norm.T @ A_norm, self.k, maxiter=10000)
      v_disim_L_rownorm = np.sqrt(np.sum(v_disim_L**2, axis=1, keepdims=True))
      v_disim_L_rownorm[v_disim_L_rownorm==0] = 1e-5
      v_disim_R_rownorm = np.sqrt(np.sum(v_disim_R**2, axis=1, keepdims=True))
      v_disim_R_rownorm[v_disim_R_rownorm==0] = 1e-5
      v_disim_L = v_disim_L / v_disim_L_rownorm
      v_disim_R = v_disim_R / v_disim_R_rownorm

      # Step 3. k-means clustering on row normalized eigenvectors
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat_L = kmeans.fit_predict(v_disim_L)
      y_hat_R = kmeans.fit_predict(v_disim_R)
      return y_hat_L, y_hat_R

    def AATATA(self):
      '''
      Left and right SVD for clustering
      '''
      Ad = self.Ad
      val_L, v_L = sp.linalg.eigsh(Ad @ Ad.T, self.k)
      val_R, v_R  = sp.linalg.eigsh(Ad.T @ Ad, self.k)
      V = np.hstack((v_L.real, v_R.real))   #shape: (N,k) row column leads to independent partition
      # V_norm = np.sqrt(np.sum(V**2, axis=1, keepdims=True))
      # V = V / V_norm
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat = kmeans.fit_predict(V)
      return y_hat


    def DSCORE(self):
      Ad = self.Ad
      k = self.k
      N = self.N
      # Step 1. Compute SVD
      val_L, v_L = sp.linalg.eigsh(Ad @ Ad.T, k)
      val_R, v_R  = sp.linalg.eigsh(Ad.T @ Ad, k)
      v_L = np.real(v_L)
      v_R = np.real(v_R)
      v_L0 = v_L[:,0]
      v_L0[v_L0==0] = 1e-5
      v_R0 = v_R[:,0]
      v_R0[v_R0==0] = 1e-5
      # Step 2. get embedding
      T = np.log(N)
      RU = np.zeros((N,k-1))
      RV = np.zeros((N,k-1))
      for i in range(k-1):
        RU[:,i] = v_L[:,i+1]/v_L0
        RV[:,i] = v_R[:,i+1]/v_R0
      # truncate the entries
      RU[RU>T] = T
      RU[RU<-T] = -T
      RV[RV>T] = T
      RV[RV<-T] = -T

      R = np.hstack((RU,RV)) #shape:N*2(k-1)

      kmeans = KMeans(n_clusters=k, n_init=10)
      y_hat = kmeans.fit_predict(R)
      # pdb.set_trace()
      return y_hat

    def SimpHerm(self):
      Ad = self.Ad
      k = self.k
      N = self.N
      d_in = np.sum(Ad, axis =1)
      d_in = np.array(d_in).flatten()
      d_out = np.sum(Ad, axis = 0)
      d_out = np.array(d_out).flatten()
      d = d_in + d_out

      w = exp(1j*2*pi/k)
      w_cog = exp(-1j*2*pi/k)
      # A_k = wAd + ConjTran(wAd)
      # L = I - D^{-1/2} A_k D^{-1/2}
      L =  sp.eye(N) - np.diag(1.0/np.sqrt(d)) @ (w*Ad + w_cog*Ad.T) @ np.diag(1.0/np.sqrt(d))
      if sp.issparse(L) is False:
        L = sp.csr_matrix(L)
      val, v = sp.linalg.eigsh(L, k)
      kmeans = KMeans(n_clusters=k, n_init=10)
      y_hat = kmeans.fit_predict(v.real)
      return y_hat

    def BibSym(self):
      Ad = self.Ad
      M = Ad@Ad.T + Ad.T@Ad
      val, v = sp.linalg.eigsh(M, self.k)
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat = kmeans.fit_predict(v.real)
      return y_hat



    def LE_SC(self, max_iter = 10, init = 'Herm', t = 0.05):
      """
        LE_SC algorithm.
        Parameters:
            Ad (numpy.ndarray): Graph adjacency matrix
            k (int): Number of clusters, in this study k always takes 2 
            max_iter (int): Maximum number of iterations
            init (str): Initialization method ('Herm', 'Sym', 'Bal')
            t (float): Tolerance for convergence
        Returns:
            numpy.ndarray: Cluster label vector
      """

      N = self.N
      Ad = self.Ad
      A = Ad + Ad.T

      count = 0
      # Initialization based on 'init'
      if init == 'Herm':
        w1, w2, w3 = 1, 0, 0
      elif init == 'Sym':
        w1, w2, w3 = 0, 1, 0
      elif init == 'Bal':
        w1, w2, w3 = 1, 1, 0

      p_, q_, eta_ = [], [], []
      par_updates = 1e3
      HH = w1 * 1j * (Ad - Ad.T) + w2 * (Ad + Ad.T) + w3 * (np.ones((N, N)) - np.eye(N))

      while count <= max_iter:
        count += 1
        #  Option 1. directly use eigendecomposition 
        HH = w1 * 1j * (Ad - Ad.T) + w2 * (Ad + Ad.T) + w3 * (np.ones((N, N)) - np.eye(N))
        y_hat = self.f_Herm2(HH)
        # Option 2. sparse efficient eigen decomposition 
        # H_sp = w1 * 1j * (Ad - Ad.T) + w2 * (Ad + Ad.T)
        # val, v = self.eff_sparse_eig(H_sp, w3, 1)
        # V_H = np.hstack((v.real, v.imag))  # shape: (N,2)
        # kmeans = KMeans(n_clusters=2, n_init=10)
        # y_hat = kmeans.fit_predict(V_H)


        # community indicator vector {0,1}^N
        y_1 = np.zeros(N)
        y_2 = np.zeros(N)
        y_1[y_hat == 1] = 1
        y_2[y_hat == 0] = 1
        len1, len2 = np.sum(y_1), np.sum(y_2)

        if len1 in (0, 1) or len2 in (0, 1):
          print('empty cluster')
          y_hat = np.random.randint(0, 2, N)
          y_1 = np.zeros(N)
          y_2 = np.zeros(N)
          y_1[y_hat == 1] = 1
          y_2[y_hat == 0] = 1

        # Count edges
        size1 = 0.5 * y_1.T @ A @ y_1
        size2 = 0.5 * y_2.T @ A @ y_2
        size12 = y_1.T @ Ad @ y_2
        size21 = y_2.T @ Ad @ y_1
        sizeG = np.ones(N).T @ Ad @ np.ones(N)

        # Drop edge count assertions for weighted graphs
        # assert sizeG == size1 + size2 + size12 + size21
        assert (N - 1) * N == len1 * (len1 - 1) + len2 * (len2 - 1) + 2 * len1 * len2

        # Calculate p, q, eta
        p_.append(2 * (size1 + size2) / (len1 * (len1 - 1) + len2 * (len2 - 1)))
        q_.append((size12 + size21) / (len1 * len2))
        eta_.append(min(size12 / (size12 + size21), size21 / (size12 + size21)))

        # Prevent zero values
        if p_[-1] == 0: p_[-1] = 1e-5
        if q_[-1] == 0: q_[-1] = 1e-5
        if eta_[-1] == 0: eta_[-1] = 1e-5

        # Update Hermitian matrix using MLE-derived formula
        w1_new = np.log((1 - eta_[-1]) / eta_[-1])
        w2_new = -np.log(4 * eta_[-1] * (1 - eta_[-1])) + 2 * np.log((p_[-1]*(1-q_[-1]))/ (q_[-1]*(1-p_[-1])))
        w3_new = 2 * np.log((1 - p_[-1]) / (1 - q_[-1]))

        update = (abs(w1 - w1_new) + abs(w2 - w2_new) + abs(w3 - w3_new)) / (abs(w1) + abs(w2) + abs(w3))
        # print(f"LE-SC iteration {count}: updates = {par_updates * 100:.1f}%")
        if update <= t:
            print(f"Iterative LE-SC converged within {count} iterations.")
            break

        w1, w2, w3 = w1_new.copy(), w2_new.copy(), w3_new.copy()


        # Output converged p, q, eta values
      return y_hat, p_, q_, eta_
  
    
    def f_Herm(self, H):
      val, v = sp.linalg.eigsh(H, k = int(ceil(self.k/2)))
      V_H = np.hstack((v.real, v.imag))  # shape: (N,k)
      kmeans = KMeans(n_clusters=self.k, n_init=10)
      y_hat = kmeans.fit_predict(V_H)
      return y_hat

    def f_Herm2(self, H):
      # Add a small regularization for numerical stability
      H = H + sp.eye(H.shape[0]) * 1e-6
      val, v = sp.linalg.eigsh(H, k = 1, ncv=min(H.shape[0], 20))
      V_H = np.hstack((v.real, v.imag))  # shape: (N,k)
      kmeans = KMeans(n_clusters=2, n_init=10)
      y_hat = kmeans.fit_predict(V_H)
      return y_hat



def k_LE_SC(Ad, k, max_iter=10, init='Herm', t=0.05):
      """
      Iteratively applies the LE_SC algorithm to obtain k clusters.
      """
      y_hat = np.zeros(Ad.shape[0], dtype=int)
      current_Ad = Ad
      current_indices = np.arange(Ad.shape[0])

      for i in range(k-1):
        cluster_alg = ClusterBy(current_Ad, 2) # always bipartition
        y_i, _, _, _ = cluster_alg.LE_SC(max_iter=max_iter, init=init, t=t)

        y_hat[current_indices[y_i==1]] = 1+2*i # label clusters from 1 to k-1
        y_hat[current_indices[y_i==0]] = 2*i

        # Find largest cluster
        largest_cluster_label = np.argmax(np.bincount(y_i))
        largest_cluster_indices = np.where(y_i == largest_cluster_label)[0]

        # Update current adjacency matrix and indices
        current_Ad = current_Ad[largest_cluster_indices,:][:,largest_cluster_indices]
        current_indices = current_indices[largest_cluster_indices]
        # y_hat[current_indices] = k -1 # assign the last cluster
      return y_hat