#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 09:52:26 2022

@author: shreyas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    file2 = open("./ProbLeft-D-Q-average", "rb")
    file3 = open("./ProbLeft-S-TSQL", "rb")
    file5 = open("./ProbLeft-TSQL", "rb")
    num_iter = 10
    arr1 = np.load(file1)
    arr2 = np.load(file2)
    arr3 = np.load(file3)
    arr5 = np.load(file5)
    
    mean1 = np.mean(arr1,axis=1)
    var1 = np.sqrt(np.var(arr1,axis=1) / num_iter)
    
    mean2 = np.mean(arr2,axis=1)
    var2 = np.sqrt(np.var(arr2,axis=1) / num_iter)
    
    mean3 = np.mean(arr3,axis=1)
    var3 = np.sqrt(np.var(arr3,axis=1) / num_iter)
    
    mean5 = np.mean(arr5,axis=1)
    var5 = np.sqrt(np.var(arr5,axis=1) / num_iter)

    x=range(100000)

    
    plt.axhline(y=0, color='grey', linestyle='dotted')
    # plt.axhline(y=0.33, color='y')
    # plt.axhline(y=-0.95, color='b')
    lines=plt.plot(x,mean1,x,mean2,x,mean5,x,mean3)
    l1,l2,l3,l4=lines
    plt.setp(lines, linestyle='--')
    plt.setp(l1, linewidth=1.5, color='r',linestyle='solid', label='QL')
    plt.setp(l2, linewidth=1.5, color='b',linestyle='dashed', label='D-Q')
    plt.setp(l3, linewidth=1.5, color='y',linestyle='dashdot', label='TSQL')
    
    plt.setp(l4, linewidth=1.5, color='g',linestyle='dashdot', label='S-TSQL')
    
    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(loc='right', bbox_to_anchor=(1, 0.5),
          ncol=1, fancybox=True, shadow=True,prop={'size': 10})
    legend.get_title().set_fontname('Times New Roman')
    legend.get_title().set_fontweight('black')

    # plt.tick_params(labelsize=20)
    # labels = ax.get_xticklabels() + ax.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]

    font2 = {'family': 'Times New Roman',
             'weight': 'black',
             'size': 10,
             }
    plt.xlabel('Number of Episodes', font2)
    plt.ylabel('$\max_a Q(a)$', font2, weight='bold')
    plt.tight_layout()
    # plt.show()
    plt.savefig('Feb29_2.pdf', dpi=600, bbox_inches='tight')
    plt.close()
