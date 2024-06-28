
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

    file1 = open("./ProbLeft-Q", "rb")
    file2 = open("./ProbLeft-D-Q-twice-average", "rb")
    file5 = open("./ProbLeft-S-TSQL", "rb")
    file9 = open("./ProbLeft-TSQL", "rb")
    
    num_iter = 1000
    
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr5 = np.load(file5)
    arr9 = np.load(file9)
    
    
    
    mean1 = np.mean(arr1,axis=1)
    var1 = np.sqrt(np.var(arr1,axis=1) / num_iter)
    mean2 = np.mean(arr2,axis=1)
    var2 = np.sqrt(np.var(arr2,axis=1) / num_iter)
    mean5 = np.mean(arr5,axis=1)
    var5 = np.sqrt(np.var(arr5,axis=1) / num_iter)
    mean9 = np.mean(arr9,axis=1)
    var9 = np.sqrt(np.var(arr9,axis=1) / num_iter)
    
    x=range(200)

    # print(var1)

    plt.errorbar(x, mean1, 2 * var1, fmt='r--', capsize=2.5, errorevery=10, markevery=10, label='QL')
    plt.errorbar(x, mean2, 2 * var2, fmt='b-o', capsize=2.5, markevery=10, errorevery=10, label='D-Q-Avg')
    plt.errorbar(x, mean5, 2 * var5, fmt='g--', capsize=2.5, markevery=10, errorevery=10,
                  label='S-TSQL')
    plt.errorbar(x, mean9, 2 * var9, fmt='y--', capsize=2.5, markevery=10, errorevery=10,
                  label='TSQL')
    
    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(loc='right', bbox_to_anchor=(1, 0.5),
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
    plt.savefig('Fig1.pdf', dpi=600, bbox_inches='tight')
    plt.close()
