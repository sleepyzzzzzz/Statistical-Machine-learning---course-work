import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
    """
    SVM hinge loss function for two class problem

    Inputs:
    - theta: A numpy vector of size d containing coefficients.
    - X: A numpy array of shape mxd 
    - y: A numpy array of shape (m,) containing training labels; +1, -1
    - C: (float) penalty factor

    Returns a tuple of:
    - loss as single float
    - gradient with respect to theta; an array of same shape as theta
    """

    m, d = X.shape
    grad = np.zeros(theta.shape)
    J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
    h = np.dot(X, theta);

    J = (1/2/m) * np.sum(np.square(theta)) + C/m * np.sum(np.maximum(np.zeros(len(h)), 1-y*h));
    grad = theta/m +  C/m * np.dot((y*h < 1).astype(float) * (-y), X)

  #############################################################################


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, C):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension d, there are K classes, and we operate on minibatches
    of m examples.

    Inputs:
    - theta: A numpy array of shape d X K containing parameters.
    - X: A numpy array of shape m X d containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
    - C: (float) penalty factor

    Returns a tuple of:
    - loss J as single float
    - gradient with respect to weights theta; an array of same shape as theta
    """

    K = theta.shape[1] # number of classes
    m = X.shape[0]     # number of examples

    J = 0.0
    dtheta = np.zeros(theta.shape) # initialize the gradient as zero
    delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Compute the loss function and store it in J.                              #
  # Do not forget the regularization term!                                    #
  # code above to compute the gradient.                                       #
  # 8-10 lines of code expected                                               #
  #############################################################################
    # for i in range(m):
    #     h = np.dot(X[i,:], theta)
    #     hy = h[y[i]]
    #     for j in range(K):
    #         if j != y[i]:
               
    #             J += max(0.0,h[j] - hy + delta)
    #         if (h[j] - hy + delta) > 0:
    #             dtheta[:, j] += X[i, :]
    #             dtheta[:, y[i]] -= X[i, :]

    # J = J/m + C * np.sum(np.square(theta))/2
    # dtheta = C * theta + dtheta/m 

    for i in range(m):
      h = np.dot(X[i, :], theta)
      h_t = y[i]
      for j in range(K):
        if j != h_t:
          margin = h[j] - h[h_t] + delta
          J += np.maximum(0, margin)
          if margin > 0:
            dtheta[:, y[i]] -= X[i, :]
            dtheta[:, j] += X[i, :]
    # grader
    J = C / m * J + 1 / (2 * m) * np.sum(np.square(theta))
    dtheta = C / m * dtheta + 1 / m * theta
    # notebook
    # J = 1 / m * J + C / (2 * m) * np.sum(np.square(theta))
    # dtheta = 1 / m * dtheta + C / m * theta

    # for i in range(m):
    #     h = np.dot(X[i,:], theta)
    #     hy = h[y[i]]
    #     for j in range(K):
    #         if j == y[i]:
    #             continue
    #         l = h[j] - hy + delta
    #         if l > 0:
    #             J += l
    #             dtheta[:, j] += X[i, :]
    #             dtheta[:, y[i]] -= X[i, :]

    # J /= m
    # dtheta /= m
    # J += 0.5 * C * np.sum(theta * theta)
    # dtheta += C * theta

    # for mm in range(m):
    #     p2 = np.dot(theta[:,y[mm]], X[mm,:])
    #     for yy in range(K):
    #         if yy != y[mm]:
    #             m2 = np.dot(theta[:,yy], X[mm,:])-p2+delta
    #             J += (max(0, m2))
    #             if m2 > 0:
    #                 dtheta[:,yy] += X[mm,:]
    #                 dtheta[:,y[mm]] -= X[mm,:]
    # J = np.sum(np.square(theta))/2/m*C + J/m
    # dtheta = theta/m*C + dtheta/m



    # num_classes = K
    # num_train = m


    # loss = 0.0
    # for i in range(num_train):
    #   scores = X[i].dot(theta)
    #   correct_class_score = scores[y[i]]
    #   diff_count = 0.0
    #   for j in range(num_classes):
    #     if j == y[i]:
    #       continue
    #     margin = scores[j] - correct_class_score + delta
    #     if margin > 0:
    #       diff_count += 1
    #       dtheta[:, j] += X[i] # gradient update
    #       loss += margin
    #   # gradient update for correct row
    #   dtheta[:, y[i]] += -diff_count * X[i]
      
    # loss /= num_train
    # print(loss)
    # dtheta /= num_train
    # dtheta += C*dtheta

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
    
    return J, dtheta


def svm_loss_vectorized(theta, X, y, C):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    J = 0.0
    dtheta = np.zeros(theta.shape) # initialize the gradient as zero
    delta = 1.0
    m = X.shape[0]
    K = theta.shape[1]
    d = theta.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in variable J.                                                     #
    # 8-10 lines of code                                                        #
    #############################################################################

    h = np.dot(X, theta)
    hy = np.choose(y, h.T).reshape(-1, 1)
    L = np.maximum(h - hy + delta, 0.0)
    L[np.arange(m), y] = 0.0
    # notebook
    # J = np.sum(L)/m + C * np.sum(np.square(theta))/2/m
    # grader
    J = C*np.sum(L)/m + np.sum(np.square(theta))/2/m
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dtheta.                                       #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    grad = (L > 0) * 1.0
    grad[range(m), y] = -np.sum(grad, axis = 1)
    # notebook
    # dtheta = np.dot(X.T, grad)/m + theta*C/m
    # grader
    dtheta = C / m * np.dot(X.T, grad) + 1 / m * theta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return J , dtheta
