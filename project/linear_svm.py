import numpy as np

def binary_svm_loss(theta, X, y, C):
    m, d = X.shape
    grad = np.zeros(theta.shape)
    J = 0
    h = np.dot(X, theta);

    J = (1/2/m) * np.sum(np.square(theta)) + C/m * np.sum(np.maximum(np.zeros(len(h)), 1-y*h));
    grad = theta/m +  C/m * np.dot((y*h < 1).astype(float) * (-y), X)
    return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, C):
    K = 7
    m = X.shape[0]

    J = 0.0
    dtheta = np.zeros(theta.shape)
    delta = 1.0
    for i in range(m):
      h = np.dot(X[i, :], theta)
      h_t = int(y[i])
      for j in range(K):
        if j != h_t:
          margin = h[j] - h[h_t] + delta
          J += np.maximum(0, margin)
          if margin > 0:
            dtheta[:, int(y[i])] -= X[i, :]
            dtheta[:, j] += X[i, :]
    J = 1 / m * J + C / (2 * m) * np.sum(np.square(theta))
    dtheta = 1 / m * dtheta + C / m * theta
    
    return J, dtheta


def svm_loss_vectorized(theta, X, y, C):
    J = 0.0
    dtheta = np.zeros(theta.shape) # initialize the gradient as zero
    delta = 1.0
    m = X.shape[0]
    K = theta.shape[1]
    d = theta.shape[0]
    h = np.dot(X, theta)
    hy = np.choose(y, h.T).reshape(-1, 1)
    L = np.maximum(h - hy + delta, 0.0)
    L[np.arange(m), y] = 0.0
    # notebook
    J = np.sum(L)/m + C * np.sum(np.square(theta))/2/m
    grad = (L > 0) * 1.0
    grad[range(m), y] = -np.sum(grad, axis = 1)
    # notebook
    dtheta = np.dot(X.T, grad)/m + theta*C/m

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return J , dtheta
