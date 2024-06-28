
"""
Created on Sat Sep  3 16:56:24 2022

@author: shreyas
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from collections import deque

if __name__ == "__main__":

    figsize = 8, 4
    figure, ax = plt.subplots(figsize=figsize)

    file1 = open("./ProbLeft-TSQL_1", "rb")
    file2 = open("./ProbLeft-TSQL_2", "rb")
    #file3 = open("./ProbLeft-check_5", "rb")
    # file4 = open("./ProbLeft-sql", "rb")
    file5 = open("./ProbLeft-TSQL_3", "rb")
    file9 = open("./ProbLeft-TSQL_4", "rb")
    
    num_iter = 1000
    
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    
    # arr3 = np.load(file3)
    # arr4 = np.load(file4)
    arr5 = np.load(file5)
    
    arr9 = np.load(file9)
    
    mean1 = np.mean(arr1,axis=1)
    var1 = np.sqrt(np.var(arr1,axis=1) / num_iter)
    mean2 = np.mean(arr2,axis=1)
    var2 = np.sqrt(np.var(arr2,axis=1) / num_iter)
    # mean3 = np.mean(arr3,axis=1)
    # var3 = np.sqrt(np.var(arr3,axis=1) / num_iter)
    # mean4 = np.mean(arr4,axis=1)
    # var4 = np.sqrt(np.var(arr4,axis=1) / num_iter)
    mean5 = np.mean(arr5,axis=1)
    var5 = np.sqrt(np.var(arr5,axis=1) / num_iter)
    
    
    mean9 = np.mean(arr9,axis=1)
    var9 = np.sqrt(np.var(arr9,axis=1) / num_iter)
    
    x=range(200)

    # print(var1)
    plt.errorbar(x, mean1, 2 * var1, fmt='r--', capsize=2.5, errorevery=10, markevery=10, label=r"TSQL-$\theta_n=\frac{10}{n+100}$")
    plt.errorbar(x, mean2, 2 * var2, fmt='b--', capsize=2.5, markevery=10, errorevery=10, label=r"TSQL-$\theta_n=\frac{(-1)^n}{n^2+10}$")
    # plt.errorbar(x, mean3, 2 * var3, fmt='k-o', capsize=2.5, markevery=10, errorevery=10, label=r"TSQL-$\theta^5_n$")
    # plt.errorbar(x, mean4, 2 * var4, fmt='y-*', capsize=2.5, markevery=10, errorevery=10,
    #                label='sql')
    plt.errorbar(x, mean5, 2 * var5, fmt='g--', capsize=2.5, markevery=10, errorevery=10,
                  label=r"TSQL-$\theta_n=\frac{-10}{n+100}$")
    plt.errorbar(x, mean9, 2 * var9, fmt='y--', capsize=2.5, markevery=10, errorevery=10,
                  label=r"TSQL-$\theta_n=\frac{1}{\sqrt{n+1}}$")
    
    
    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(loc='right', bbox_to_anchor=(1, 0.8),
          ncol=1, fancybox=True, shadow=True,prop={'size': 10})
    legend.get_title().set_fontname('DejaVu Sans')
    legend.get_title().set_fontweight('black')

    plt.tick_params(labelsize=10)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font2 = {'family': 'DejaVu Sans',
             'weight': 'black',
             'size': 10,
             }
    plt.xlabel('Number of Episodes', font2)
    plt.ylabel('Probability of a left action', font2)
    plt.tight_layout()
    plt.savefig('TSQL-theta.pdf', dpi=600, bbox_inches='tight')
    plt.close()
