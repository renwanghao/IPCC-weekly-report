import numpy as np
import scipy.linalg as linalg
import random


# quantum state
Zero = np.array([[1.0],
                 [0.0]])
One = np.array([[0.0],
                [1.0]])

P0 = np.dot(Zero, Zero.T)
P1 = np.dot(One, One.T)
# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1],
              [1, 0]])  #: Pauli-X matrix
Y = np.array([[0, -1j],
              [1j, 0]])  #: Pauli-Y matrix
Z = np.array([[1, 0],
              [0, -1]])  #: Pauli-Z matrix
Hadamard = 1. / np.sqrt(2) * np.array([[1, 1],
                                       [1, -1]])  #: Hadamard gate


# Two qubit gates
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])  #: CNOT gate = CX
CNOT21 = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]])  #: CNOT21 gate is 2 controlled 1
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
# CSWAP  用到的时候再写


def Identity(size):
    # 生成一个2的size次方单位阵
    matrix = 1
    for i in range(1, size + 1):
        matrix = np.kron(matrix, I)
        # kronecker product.张量积
    return matrix


def RX(param):  # 旋转门RX
    result = linalg.expm(-1J / 2 * param * X)
    return result


def RY(param):
    result = linalg.expm(-1J / 2 * param * X)
    return result


def RZ(param):
    result = linalg.expm(-1J / 2 * param * X)
    return result


def RZYZ(param1, param2, param3):
    result = np.dot(RZ(param1), np.dot(RY(param2), RZ(param3)))
    return result

# tools


def Ndot44(*args):
    """Calculate a matrix product over a variable number of inputs"""
    result = np.kron(I, I)
    for op in args:
        result = np.dot(result, op)
    return result

# t = Ndot44(CNOT, CNOT21, CNOT)
# print(t)


def random_ListTheta(size):
    # 随机生成一串长度为 size 的list
    li = []

    for a in range(size):
        i = random.random()
        li.append(np.pi * i)
    return li


def zero_ListTheta(size):
    li = []
    for a in range(size):
        i = 0
        li.append(i)
    return li


# b = random_ListTheta(15)
# print(b)


def Vtheta(lst):
    # 1 传出参数theta的序列，2 输出矩阵表示。
    # 传入参数lst(长度必须是15)， 进行计算，算出矩阵表达a5，传出list。
    if len(lst) is not 15:
        print('thetalist is error,thetalist should be 15')
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = lst[0: len(lst)]
    a1 = np.kron(RZYZ(v1, v2, v3), RZYZ(v4, v5, v6))
    a2 = np.kron(RZ(v7), RY(v8))
    a3 = np.kron(I, RY(v9))
    a4 = np.kron(RZYZ(v10, v11, v12), RZYZ(v13, v14, v15))
    a5 = Ndot44(a4, CNOT21, a3, CNOT, a2, CNOT21, a1)
    lst1 = lst
    return a5, lst1

# b2 = ListThera(45)
# 这里想写两个量子比特以上的情况
# def Vtheta_2(lst):
#     if len(lst) is not 30:
#         print('thetalist is error,thetalist should be 30')
#     v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30 = lst[0: len(
#         lst)]


def Vtheta2(lst):
    if len(lst) is not 30:
        print('thetalist is error,thetalist should be 30 for input')
    lst1 = lst[0: 14]
    lst2 = lst[15: len(lst)]
    U1, lst1 = Vtheta(lst1)
    U2, lst2 = Vtheta(lst2)
    matrix_rep = np.dot(np.kron(I, U2), np.kron(U1, I))
    lst = lst1 + lst2
    if len(lst) is not 30:
        print('thetalist is error,thetalist should be 30 for output')
    return matrix_rep, lst


# c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# d, lst2 = Vtheta(b)
# print(d)  # 这里是测试v0-v15全是0的时候，Vthera就是一个swap。
# print(lst2)


def NKron(*args):
    """Calculate a Kronecker product over a variable number of inputs"""
    result = np.array([[1.0]])
    for op in args:
        result = np.kron(result, op)
    return result


class Quantum_Circuit:
    def __init__(self, size, name, lst):
        self.size = size
        self.name = name
        self.param = lst

    def Vtheta1(self):
        # 1 传出参数theta的序列，2 输出矩阵表示。
        # 传入参数lst(长度必须是15)， 进行计算，算出矩阵表达a5，传出list。
        if len(self.param) is not 15:
            print('thetalist is error,thetalist should be 15')
        v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 = self.param[0: len(
            self.param)]
        a1 = np.kron(RZYZ(v1, v2, v3), RZYZ(v4, v5, v6))
        a2 = np.kron(RZ(v7), RY(v8))
        a3 = np.kron(I, RY(v9))
        a4 = np.kron(RZYZ(v10, v11, v12), RZYZ(v13, v14, v15))
        matrix_rep = Ndot44(a4, CNOT21, a3, CNOT, a2, CNOT21, a1)
        lst1 = self.param
        return matrix_rep, lst1

    def Vtheta2(self):
        if len(self.param) is not 30:
            print('thetalist is error,thetalist should be 30 for input')
        a = len(self.param)
        lst1 = self.param[0: 15]
        lst2 = self.param[15: len(self.param)]
        lst = self.param[0: len(self.param)]
        U1, lst1 = Vtheta(lst1)
        U2, lst2 = Vtheta(lst2)
        matrix_rep = np.dot(np.kron(I, U2), np.kron(U1, I))
        lst3 = lst1 + lst2
        if len(lst) is not 30:
            print('thetalist is error,thetalist should be 30 for output')
        return matrix_rep, lst3

    # def gen_mat(self):
    #     return matrix_rep
