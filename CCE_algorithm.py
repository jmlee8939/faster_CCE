import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import time
from numpy.random import rand


def CCE(connectivity, max_iteration, targetcenternumber=1, check=10):
    """ Run CCE algorithm 
    
    Args :
        connectivity : connectivity matrix of data
        max_iteration : max iteration (max order k) for CCE algoritm
        targetcenternumber : the target number of centers 
        (CCE will stop when find fewer centers then targetcenternuber)
        check : the interval of iterations which show CCE results
       
    Returns :
        tuple of results (n_center, center_id, label, n_cut)
            n_center : the number of centers from 1 order to last order K (list)
            center_id : index of centers in last order K (list)
            label : labels of data in last order K (array)
            n_cut : n_cut of results from 1 order to last order K (list)
            
    """
    
    def Ncut(P, center_id, label):
        """ Calculate Ncut
        
        Args :
            P : connectivity matrix (array)
            center_id : center index (list)
            label : labels of data
            
        Returns :
            Ncut value (numeric)
        """
        ncut = 0
        for k in range(len(center_id)):
            l = center_id[k]
            ncut = ncut + np.sum(P[label == l].T[label != l]) / len(label[label == l])
        return ncut

    # base parameter
    n_center = list()
    temp_n_center = list()
    n_cut = list()
    P = connectivity
    n, m = P.shape
    
    P_0 = P
    center_id = list()
    for i in range(max_iteration):
        center_id = list()
        P_diag = np.diag(P)
        P_res = P - np.diag(P_diag)
        # find center
        for j in range(n):
            if P_diag[j] > np.max(P_res[j, :]):
                center_id.append(j)

        # number of center(cluster)
        temp_n_center.append(len(center_id))
        n_center.append(min(temp_n_center))

        # data labeling
        P_c = P*(1/np.diag(P))
        P_c = P_c[:, center_id]
        label = np.array(list(map(lambda x: center_id[x], np.argmax(P_c, 1))))

        # ncut
        n_cut.append(Ncut(P, center_id, label))
        
        if i % check == 0:
            print('order of connectivity matrix = %d' % (i + 1))
            print('number of cluster center = %d' % (min(temp_n_center)))
            print('...')

        if len(center_id) <= 1:
            print('one cluster center')
            print('order %d' % (i + 1))
            print(center_id)
            return n_center, center_id, label, n_cut

        if len(center_id) <= targetcenternumber:
            print('less then %d cluster center' % (targetcenternumber))
            print('order %d' % (i + 1))
            print(center_id)
            return n_center, center_id, label, n_cut
        
        # next order connectivity  
        P = np.dot(P, P_0)
    return n_center, center_id, label, n_cut


def connectivity_matrix(X, sigma):
    """ Calculate ergodic connectivity with gaussian kernel
    
    Args :
        X : data (array)
        sigma : standard deviation (numeric)
        
    Returns :
        ergodic connectivity matrix (array)
    """
    
    n, p = X.shape
    D = -2 * np.dot(X, X.T) + np.tile(np.sum(X ** 2, 1), 
                                      (n, 1)).T + np.tile(np.sum(X ** 2, 1), (n, 1))
    S = np.exp(-D / (2 * sigma ** 2))
    S = S / np.tile(np.sum(S, 1), (n, 1)).T
    return S

def enhanced_connectivity_matrix(X, sigma, epsilon):
    """ Calculate enhanced connectivity with gaussian kernel
    
    Args :
        X : data (array)
        sigma : standard deviation (numeric)
        epsilon : parameter for control enhanced merging effect 
        
    Returns :
        enhanced connectivity matrix (array)
    """
    n,p = X.shape
    D = -2*np.dot(X,X.T)+np.tile(np.sum(X**2,1),(n,1)).T+np.tile(np.sum(X**2,1),(n,1))
    S = np.exp(-D/(2*sigma**2))
    S[S <= epsilon] = epsilon
    S = S / np.tile(np.sum(S,1),(n,1)).T
    S_hat = S
    return S_hat
