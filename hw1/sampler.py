#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:34:29 2021

@author: xiongn
"""

import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        z = self.mu + np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2) * self.sigma
        return z

    def plot_his(self):
        z = []
        for i in range(5000):
            z.append(self.sample())
        plt.hist(z)
        plt.show()

    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu = Mu
        self.Sigma = Sigma

    def sample(self):
        L = np.linalg.cholesky(self.Sigma)
        u = []
        for i in range(self.Mu.size):
            u.append(UnivariateNormal(0, 1).sample())
        z = self.Mu + np.dot(L, u)
        return z

    def plot_his(self):
        x = []
        y = []
        for i in range(5000):
            z = self.sample()
            x.append(z[0])
            y.append(z[1])
        plt.scatter(x, y)
        plt.show()
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        self.ap = ap

    def sample(self):
        cum = np.cumsum(self.ap)
        u = np.random.uniform(0, 1)
        for i in range(len(self.ap)):
            if u < cum[i]:
                return i

    def plot_his(self):
        z = []
        for i in range(5000):
            z.append(self.sample())
        plt.hist(z)
        plt.show()


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        self.ap = ap
        self.pm = pm

    def sample(self):
        rand = np.random.uniform(0, 1);
        pts = self.pm[int(rand*len(self.ap))].sample();
        return pts;
    
    def plot(self):
        x = []
        y = []
        for i in range(5000):
            z = self.sample()
            x.append(z[0])
            y.append(z[1])
        plt.scatter(x, y)
        plt.show()        
        

if __name__ == '__main__':
    U = UnivariateNormal(1, 1)
    U.plot_his()
    C = Categorical(np.array([0.1, 0.1, 0.3, 0.3, 0.2]))
    C.plot_his()
    M = MultiVariateNormal(np.array([1, 1]), np.array([[1., 0.5], [0.5, 1.]]))
    M.plot_his()
    cov = np.array([[1., 0], [0, 1.]])
    pm = [MultiVariateNormal(np.array([1, 1]), cov),MultiVariateNormal(np.array([-1, 1]), cov),MultiVariateNormal(np.array([1, -1]), cov),
          MultiVariateNormal(np.array([-1, -1]), cov)]
    pm=tuple(pm)
    Mix = MixtureModel(np.array([.25, .25, .25, .25]), pm)
    Mix.plot()
    
    
    