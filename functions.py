# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

FORCE_DOMAIN = 7

def Display_membership_functions(title, *functions):
    fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
    for index, tup in enumerate(functions):
        values = []
        for x in tup[0]:
            values.append(tup[1](x))        
        
        ax0.plot(tup[0], values, tup[2], linewidth=1.5, label=tup[3])
    ax0.legend()
    ax0.set_title(title)
    plt.tight_layout()
    
def Compute_weighted_integral(func):
    exs = np.linspace(-FORCE_DOMAIN, FORCE_DOMAIN, 101)
    return sum([x * func(x) for x in exs]) / sum([func(x) for x in exs])