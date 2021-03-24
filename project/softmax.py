import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - theta: d x K parameter matrix. Each column is a coefficient vector for class k
    - X: m x d array of data. Data are d-dimensional rows.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to parameter matrix theta, an array of same size as theta
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    #############################################################################
    theta_x = np.dot(X, theta);
    # regularize theta
    theta_x_reg = theta_x - np.max(theta_x, axis = 1).reshape(-1,1);
    
    exp_theta = np.exp(theta_x_reg);
    P = exp_theta/(np.sum(exp_theta,axis=1).reshape(-1,1));
    for i in range(m):
        for j in range(len(theta[0])):
            if(y[i] == j):
                l = np.log(P[i,j]);
                J += l
    J = -J/m;
    for i in range(dim):
        for j in range(len(theta[0])):
            J += reg / 2 / m * np.square(theta[i,j]);
       
    for i in range(len(theta[0])):
        for j in range(m):
            if (y[j] == i):
                I = 1;
            else:
                I = 0;
                    
            grad[:, i] += (X[j,:] * (I - P[j,i])) 
        grad[:,i] = -grad[:,i]/m
        grad[:,i] += reg/m * theta[:,i]
    
    

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################
    theta_x = np.dot(X, theta);
    # regularize theta
    theta_x_reg = theta_x - np.max(theta_x, axis = 1).reshape(-1,1);
    
    exp_theta = np.exp(theta_x_reg);
    P = exp_theta/(np.sum(exp_theta,axis=1).reshape(-1,1));
    I = np.zeros([m, len(theta[0])]);
    ind = np.linspace(0,m-1,m,dtype='int')
    I[ind , y] = 1;
    #print(i)
    J = -1/m * np.sum(np.multiply(I, np.log(P))) + reg/2/m * np.sum(np.square(theta));
    
    grad = -1/m * np.dot(X.T, I - P) + reg/m * theta

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad
