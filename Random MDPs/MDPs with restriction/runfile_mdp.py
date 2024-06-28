#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:35:05 2022

@author: Shreyas S R
"""



import numpy as np
import mdptoolbox, mdptoolbox.example
import random
from numpy.random import seed



#np.random.seed(0)
#P, R = mdptoolbox.example.forest()
s = 10
a= 5
discount = 0.6
q_count = 0
sor_q_count = 0
policy_count = 0
sor_policy_count = 0
percentage_count = 0

episodes = 100
iterations = 100000




qlearningerr = np.zeros((episodes,1))
TSQLerr= np.zeros((episodes,1))
STSQLerr= np.zeros((episodes,1))
sorqlearningerr=np.zeros((episodes,1))
doubleqlearningerr = np.zeros((episodes,1))
MF_doubleqlearningerr = np.zeros((episodes,1))
MF_SOR_QLearningerr =  np.zeros((episodes,1))
for count in range(episodes):
    
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)
    
    P, R = mdptoolbox.example.rand(s, a)
    
    
    
    # Value Iteration
    
    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.000001)
    vi.run()
    
    
    # print('************************************')
    #Q-learning
    
    ql = mdptoolbox.mdp.QLearning(P, R, discount,n_iter=iterations)
    
    ql.run()
    
    qlearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql.V)))
    
    
    STSQL = mdptoolbox.mdp.STSQL(P, R, discount,n_iter=iterations)
    
    STSQL.run()
    
    STSQLerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(STSQL.V)))
    
    #SOR Q-learning
    
    ql2 = mdptoolbox.mdp.SOR_QLearning(P, R, discount,n_iter=iterations)
    ql2.run()
    sorqlearningerr[count]=np.linalg.norm((np.asarray(vi.V) - np.asarray(ql2.V)))
    
    
    ql5 = mdptoolbox.mdp.Double_QLearning(P, R, discount,n_iter=iterations)
    ql5.run()
    doubleqlearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql5.V)))
    
    
    ql10 = mdptoolbox.mdp.TSQL(P, R, discount,n_iter=iterations)
    ql10.run()
    TSQLerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql10.V)))
    
    # ql11 = mdptoolbox.mdp.MF_SOR_QLearning(P, R, discount,n_iter=iterations)
    # ql11.run()
    # MF_SOR_QLearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql11.V)))
    
    # ql12 = mdptoolbox.mdp.MF_Double_QLearning(P, R, discount,n_iter=iterations)
    # ql12.run()
    # MF_doubleqlearningerr[count] = np.linalg.norm((np.asarray(vi.V) - np.asarray(ql12.V)))
    
    
    
print("Q-Learning Error:",np.mean(qlearningerr))

print("TSQL Error:",np.mean(TSQLerr))

print("S-TSQL Error:",np.mean(STSQLerr))
print("SOR Q-learning Error:",np.mean(sorqlearningerr))

print("Double Q-Learning Error:",np.mean(doubleqlearningerr))
# print("Model-free SOR Q-learning",np.mean(MF_SOR_QLearningerr))
# print("Model-fre Double Q-learning error", np.mean(MF_doubleqlearningerr))