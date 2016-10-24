'''Module for gamma functions'''

import math

def high_constant_gamma(t):
    return 1

def low_constant_gamma(t):
    return 0
    
def high_exponential_gamma(t):
    return math.pow(.8, t)
    
def med_exponential_gamma(t):
    return math.pow(.5, t)

def low_exponential_gamma(t):
    return math.pow(.25, t)
    

