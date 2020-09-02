# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

FORCE_RANGE = 12
NEGATIVE = 'Negative'
ZERO = 'Zero'
POSITIVE = 'Positive'

def Display_membership_functions(title, value_range, *functions):
    
    domain = np.linspace(-value_range, value_range, 101)
    fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
    
    tup = Memebership_display_tuples(functions)
    
    for el in tup:
        values = []
        for x in domain:
            values.append(el[0](x))        
        ax0.plot(domain, values, el[1], linewidth=1.5, label=el[2])
    ax0.legend()
    ax0.set_title(title)
    plt.tight_layout()
    
def Compute_weighted_integral_force(func):
    exs = np.linspace(-FORCE_RANGE, FORCE_RANGE, 51)
    weights = sum([func(x) for x in exs])
    value = sum([x * func(x) for x in exs])
    # nie wiem czy ta suma w ogóle powinna mieć kiedykolwiek zerową sumę wag :/
    if weights == 0:    
        if value > 0:
            return FORCE_RANGE
        else:
            return -FORCE_RANGE
        
    return  value / weights

def Memebership_display_tuples(functions):
    return ((functions[0], 'r', 'Negative'),
            (functions[1], 'g', 'Zero'),
            (functions[2], 'b', 'Positive'))

def Generic_membership_functions(value_range):
    neg_func = lambda x: min(max(-x, 0.0) / value_range, 1.0)
    zer_func = lambda x: max(-abs(x) / value_range + 1, 0.0)
    pos_func = lambda x: min(max(x, 0.0) / value_range, 1.0)
    return {NEGATIVE : neg_func, ZERO : zer_func, POSITIVE : pos_func}



















