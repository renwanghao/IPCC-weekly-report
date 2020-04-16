import numpy as np


def Ndot44(*args):
    """Calculate a matrix product over a variable number of inputs"""
    result = np.kron(I, I)
    for op in args:
        result = np.dot(result, op)
    return result

t = Ndot44(CNOT, CNOT21, CNOT)
print(t)