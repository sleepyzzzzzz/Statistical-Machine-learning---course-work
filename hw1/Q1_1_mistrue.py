# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 02:48:21 2021

@author: Neng Xiong
"""

from numpy import *
from matplotlib.pylab import *

#from pypr.clustering import *




def sample_gaussian_mixture(centroids, ccov, mc = None, samples = 1):
    """
    Draw samples from a Mixture of Gaussians (MoG)

    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.

    Returns
    -------
    X : 2d np array
         A matrix with samples rows, and input dimension columns.

    Examples
    --------
    ::

        from pypr.clustering import *
        from numpy import *
        centroids=[array([10,10])]
        ccov=[array([[1,0],[0,1]])]
        samples = 10
        gmm.sample_gaussian_mixture(centroids, ccov, samples=samples)

    """
    cc = centroids
    D = len(cc[0]) # Determin dimensionality
    
    # Check if inputs are ok:
    K = len(cc)
    if mc is None: # Default equally likely clusters
        mc = np.ones(K) / K
    if len(ccov) != K:
        raise ValueError("centroids and ccov must contain the same number" +\
            "of elements.")
    if len(mc) != K:
        raise ValueError("centroids and mc must contain the same number" +\
            "of elements.")

    # Check if the mixing coefficients sum to one:
    EPS = 1E-15
    if np.abs(1-np.sum(mc)) > EPS:
        raise ValueError("The sum of mc must be 1.0")

    # Cluster selection
    cs_mc = np.cumsum(mc)
    cs_mc = np.concatenate(([0], cs_mc))
    sel_idx = np.random.rand(samples)

    # Draw samples
    res = np.zeros((samples, D))
    for k in range(K):
        idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
        ksamples = np.sum(idx)
        drawn_samples = np.random.multivariate_normal(\
            cc[k], ccov[k], ksamples)
        res[idx,:] = drawn_samples
    return res

mc = [0.25, 0.25, 0.25, 0.25] # Mixing coefficients
centroids = [ array([1,1]), array([1,-1]), array([-1,1]), array([-1,-1]) ]
ccov = [ diag((1,1)), diag((1,1)), diag((1,1)),diag((1,1)) ]

X = sample_gaussian_mixture(centroids, ccov, mc, samples=2000)
dis = X - [0.1,0.2]
dis_array = np.sqrt(dis[:,0]**2 + dis[:,1]**2)
print(len(dis_array[dis_array<=1])/2000)


plot(X[:,0], X[:,1], '.')

