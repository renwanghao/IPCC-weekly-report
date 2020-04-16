# 自己写量子噪声


import numpy as np
import scipy as sp
import scipy.linalg

Id = np.eye(2)
Zero = np.array([[1.0],
                 [0.0]])
One = np.array([[0.0],
                [1.0]])
# 定义0态和1态

P0 = np.dot(Zero, Zero.T)
P1 = np.dot(One, One.T)
# 定义外积00和11

X = np.array([[0, 1],
              [1, 0]])
# 定义X门




def NKron(*args):
    """Calculate a Kronecker product over a variable number of inputs"""
    result = np.array([[1.0]])
    for op in args:
        result = np.kron(result, op)
    return result


# print(NKron(P0, Id))
# print(NKron(P1, X))
# # print(NKron(P0, Id) + NKron(P1, X))
# CNOT = NKron(P0, Id) + NKron(P1, X)
# CONTT = NKron(Id, P0) + NKron(X, P1)
# SWAP = np.dot(np.dot(CNOT, CONTT), CNOT)
# print(CNOT)
# print(CONTT)
# print(SWAP)


def NormalizeState(state): return state / sp.linalg.norm(state)
# 归一化


def main():
    p = np.random.rand()
    print(p)

    CNOT = NKron(P0, Id) + NKron(P1, X)
    CONTT = NKron(Id, P0) + NKron(X, P1)
    SWAP = np.dot(np.dot(CNOT, CONTT), CNOT)

    if (np.random.rand() < p):
        # 小于随机概率P0，视为以P概率执行
        Result = NormalizeState(NKron(Id, Id, P0))
        print(0)
    else:
        # 不小于P0。视为1-P概率执行
        Result = NormalizeState(NKron(SWAP, P1))
        print(1)

    DepolarizingChannel = Result
    # DepolarizingChannel = NKron(Id, Id, P0) + NKron(SWAP, P1)
    print(DepolarizingChannel)


if __name__ == '__main__':
    main()
