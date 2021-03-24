import numpy as np
import utils 
import math
import scipy 
from scipy import optimize
import random
from scipy.special import xlogy

class RegLogisticRegressor:
    def __init__(self):
        self.theta = None

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def train(self,X,y,reg=1e-5,num_iters=1000,norm=True):
        num_train,dim = X.shape
        if norm:
            X_without_1s = X[:,1:]
            X_norm, mu, sigma = utils.std_features(X_without_1s)
            XX = np.vstack([np.ones((X_norm.shape[0],)),X_norm.T]).T
        else:
            XX = X
        theta = np.zeros((dim, 7))
        print (theta)
        theta_opt_norm = scipy.optimize.fmin_bfgs(self.loss, theta.reshape((dim, 7)), fprime = self.grad_loss, args=(XX,y,reg),maxiter=num_iters)
        if norm:
            theta_opt = np.zeros(theta_opt_norm.shape)
            theta_opt[1:] = theta_opt_norm[1:]/sigma
            theta_opt[0] = theta_opt_norm[0] - np.dot(theta_opt_norm[1:],mu/sigma)
        else:
            theta_opt = theta_opt_norm
        return theta_opt

    def loss(self, *args):
        theta,X,y,reg = args
        m,dim = X.shape
        theta = theta.reshape((dim, 7))
        J = 0
        htheta = self.sigmoid(np.dot(X,theta))
        J = np.sum((-y).reshape(-1, 1) * np.log(htheta) - (1-y).reshape(-1, 1) * np.log(1-htheta))/m + (np.sum(np.square(theta[1:len(theta)]))) * reg/2/m
        if (J == float("inf")):
            J = float("nan")
        return J

    def grad_loss(self, *args):
        theta,X,y,reg = args
        m,dim = X.shape
        theta = theta.reshape((dim, 7))
        grad = np.zeros((dim, 7))
        grad = np.sum(np.dot(X.T, self.sigmoid(np.dot(X,theta))-y.reshape(-1,1)),axis=0)/m + theta * reg/m
        grad[0] = grad[0] - theta[0] * reg/m
        return grad
        
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred = (np.dot(X, self.theta)>0).astype(int)
        return y_pred




# import numpy as np
# import utils 
# import math
# import scipy 
# from scipy import optimize
# import random
# from scipy.special import xlogy

# class LogisticRegressor:

#     def __init__(self):
#         self.theta = None

#     def sigmoid(self, x):
#         return (1 / (1 + np.exp(-x)))


#     def train(self,X,y,num_iters=1000):
#         num_train,dim = X.shape
#         X_without_1s = X[:,1:]
#         X_norm, mu, sigma = utils.std_features(X_without_1s)

#         # add the ones back and assemble the XX matrix for training

#         XX = np.vstack([np.ones((X_norm.shape[0],)),X_norm.T]).T
#         theta = np.zeros((dim,))

#         # Run scipy's fmin algorithm to run gradient descent

#         theta_opt_norm = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(XX,y),maxiter=num_iters)

#         # convert theta back to work with original X
#         theta_opt = np.zeros(theta_opt_norm.shape)
#         theta_opt[1:] = theta_opt_norm[1:]/sigma
#         theta_opt[0] = theta_opt_norm[0] - np.dot(theta_opt_norm[1:],mu/sigma)


#         return theta_opt

#     def loss(self, *args):
#         theta,X,y = args
#         m,dim = X.shape
#         J = 0
#         htheta = self.sigmoid(np.dot(X,theta))
#         J = np.sum((-y) * np.log(htheta) - (1-y) * np.log(1-htheta))/m    
#         return J

#     def grad_loss(self, *args):
#         theta,X,y = args
#         m,dim = X.shape
#         grad = np.zeros((dim,))
#         grad = np.sum(X * (self.sigmoid(np.dot(X,theta))-y)[:,None],axis=0)/m
#         return grad
        

#     def predict(self, X):
#         y_pred = np.zeros(X.shape[0])
#         y_pred = (np.dot(X, self.theta)>0).astype(int)
#         return y_pred