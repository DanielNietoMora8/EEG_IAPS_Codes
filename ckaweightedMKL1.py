#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
from scipy.stats import zscore
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix


# In[2]:


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def ckaweightedK(K,l):
    L = np.matmul(l,l.T)

    P = K.shape[0]
    N = K.shape[2]

    U = np.zeros([N,N], dtype = float)
    U = np.eye(N) - (np.matmul(np.ones([N,1]) , np.ones([1,N])))*(1/N)

    #Centering the Labels kernel
    L = np.matmul(np.matmul(U,L),U)

    #Centering the features kernels
    Kcc = np.zeros_like(K,dtype = float)
    a = np.zeros(P,dtype = float)
    aux3 = np.trace(np.matmul(L,L))
    for i in range(P):
        Kcc[i,:,:] = np.matmul(np.matmul(U,np.squeeze(K[i,:,:])),U)
        aux1 = np.trace(np.matmul(np.squeeze(Kcc[i,:,:]),L))
        aux2 = np.trace(np.matmul(np.squeeze(Kcc[i,:,:]),np.squeeze(Kcc[i,:,:])))
        a[i] = aux1 / np.sqrt(aux2*aux3)
 

 #Compunting fobenious products between pairs of kernels
    M = np.eye(P,dtype=float)
    for i in range(P):
        for j in range(i+1,P):
            aux = np.trace(np.matmul(np.squeeze(K[i,:,:]),np.squeeze(K[j,:,:])))
            aux = aux / np.sqrt(np.trace(np.matmul(np.squeeze(K[i,:,:]),np.squeeze(K[i,:,:]))) * np.trace(np.matmul(np.squeeze(K[j,:,:]),np.squeeze(K[j,:,:]))))
            M[i,j] = aux
            M[j,i] = aux


    #Solving the quadratic optimization problem        
    A = np.ones(P)
    A = matrix(A,(1,P),'d')
    b = matrix(1,(1,1),'d')
    G = -np.eye(P)
    h = np.zeros(P)
    w = cvxopt_solve_qp(P = 2*M, q = -2*a, G = G, h = h, A = A, b = b)
    eta = w / np.linalg.norm(w)
    return w


# In[4]:


#Multiple Kernel Learning function
#Inputs: A np array of size (NxNxP) where N is the number of features and P is the number of Kernels
#L: a vector of labels for creating the "label kernel yy'"

Ko = np.random.rand(10,39,39)
K = np.zeros_like(Ko)
for i in range(K.shape[0]):
    K[i,:,:] = np.matmul(np.squeeze(Ko[i,:,:]),np.squeeze(Ko[i,:,:]).T)
    
l = np.ones([39,1])
l[0:20] = -1
print(K.shape,l.shape)
w = ckaweightedK(K,l)


# In[ ]:




