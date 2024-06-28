#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:35:05 2022

@author: jitendra
"""


# coding: utf-8

# In[ ]:


import numpy as np
import mdptoolbox, mdptoolbox.example
import random
from numpy.random import seed



#np.random.seed(0)
#P, R = mdptoolbox.example.forest()

s = 10
a= 5
discount = 0.6



episodes = 100
iterations = 10000


qlearningerr = np.zeros((episodes,1))

TSQLerr= np.zeros((episodes,1))
                  
STSQLerr = np.zeros((episodes,1))

doubleqlearningerr = np.zeros((episodes,1))

for count in range(episodes):
    
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)
    
    P, R = mdptoolbox.example.rand(s, a)
    
    
    # Value Iteration

    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.000001)
    vi.run()
    
    
    
    # Q-learning
    
    ql = mdptoolbox.mdp.QLearning(P, R, discount,n_iter=iterations)
    
    ql.run()
    
    qlearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V)))
    
    # S-TSQL
    
    STSQL = mdptoolbox.mdp.STSQL(P, R, discount,n_iter=iterations)
    
    STSQL.run()
    
    STSQLerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(STSQL.V)))
    
    # Double Q-learning
    
    ql5 = mdptoolbox.mdp.Double_QLearning(P, R, discount,n_iter=iterations)
    
    ql5.run()
    
    doubleqlearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql5.V)))
    
    # S-TSQL
    
    ql10 = mdptoolbox.mdp.TSQL(P, R, discount,n_iter=iterations)
    
    ql10.run()
    
    TSQLerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql10.V)))
    
    
    
    
    
    
print("Average Q-Learning Error:",np.mean(qlearningerr))

print("Average TSQL Error:",np.mean(TSQLerr))

print("Average S-TSQL Q-Learning Error:", np.mean(STSQLerr))

print("Average Double Q-Learning Error:",np.mean(doubleqlearningerr))

