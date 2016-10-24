'''Module for epsilon functions'''

def rational_epsilon(t):
    return 1.0/t if t != 0 else 1
    
def high_constant_epsilon(t):
    return .4
    
def med_constant_epsilon(t):
    return .2
    
def low_constant_epsilon(t):
    return .1
