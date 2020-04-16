# https://blog.csdn.net/m0_37622530/article/details/88875073
# 怎么写量子线路

import numpy.random
import numpy as np
import scipy as sp
import scipy.linalg

Zero = np.array([[1.0],
                 [0.0]])
One = np.array([[0.0],
                [1.0]])


print(Zero)
print(One)


def NormalizeState(state): return state / sp.linalg.norm(state)


# 用lambda来的用法来表示一种函数，分号之前是参数，分号之后是所做的事情。
# sp。linalg。norm 这是求范数的方程。
# 所以这个归一化态函数，NormalizeState是态的矩阵，除以他们的归一化函数
Plus = NormalizeState(Zero + One)
print(Plus)

Hadamard = 1. / np.sqrt(2) * np.array([[1, 1],
                                       [1, -1]])

NewState = np.dot(Hadamard, Zero)
print(NewState)

ZeroZero = np.kron(Zero, Zero)
OneOne = np.kron(One, One)
PlusPlus = np.kron(Plus, Plus)
# 在程序中用kron（the Kronecker product）来作为tensor product。
print(ZeroZero)
print(OneOne)
print(PlusPlus)


def NKron(*args):
    """Calculate a Kronecker product over a variable number of inputs"""
    result = np.array([[1.0]])
    for op in args:
        result = np.kron(result, op)
    return result


# 多个量子态tensor
FiveQubitState = NKron(One, Zero, One, Zero, One)

print(FiveQubitState)

Id = np.eye(2)
HadamardZeroOnFive = NKron(Hadamard, Id, Id, Id, Id)
NewState = np.dot(HadamardZeroOnFive, FiveQubitState)

print(NewState)

P0 = np.dot(Zero, Zero.T)
P1 = np.dot(One, One.T)
X = np.array([[0, 1],
              [1, 0]])

CNOT03 = NKron(P0, Id, Id, Id, Id) + NKron(P1, Id, Id, X, Id)
# 问题：为啥是加起来？引申问题，量子线路怎么变化为矩阵。
NewState = np.dot(CNOT03, FiveQubitState)

print(NewState)


CatState = NormalizeState(ZeroZero + OneOne)
RhoCatState = np.dot(CatState, CatState.T)
# 这里要测试的状态就是 catstate，
# Find probability of measuring 0 on qubit 0
Prob0 = np.trace(np.dot(NKron(P0, Id), RhoCatState))

# Simulate measurement of qubit 0
if (np.random.rand() < Prob0):
    # Measured 0 on Qubit 0
    Result = 0
    ResultState = NormalizeState(np.dot(NKron(P0, Id), CatState))
else:
    # Measured 1 on Qubit 1
    Result = 1
    ResultState = NormalizeState(np.dot(NKron(P1, Id), CatState))

print(Prob0)
print("Qubit 0 Measurement Result: {}".format(Result))

print("Post-Measurement State:")

print(ResultState)

# Two qubit gates
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])  #: CNOT gate = CX
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])  #: CZ gate
CY = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, -1J],
               [0, 0, -1J, 0]])  #: CY gate i怎么输入
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])  #: SWAP gate
a = np.dot(np.kron(Id, SWAP), np.kron(CNOT, Id))
print(a)
