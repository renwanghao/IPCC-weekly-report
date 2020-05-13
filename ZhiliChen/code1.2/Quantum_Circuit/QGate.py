import numpy as np
from scipy import linalg as lin

Zero = np.array([[1.0],[0.0]])
One = np.array([[0.0],[1.0]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)
P0 = np.dot(Zero, Zero.T)
P1 = np.dot(One, One.T)

class CNOT:
    def __init__(self, i, j, n):
        '''
        :param i: control qubit
        :param j: act on jth qubit
        :param n: qubits quantity
        '''
        self.control_bit = i
        self.act_bit = j
        Cx1 = np.array([[1.0]])
        Cx2 = np.array([[1.0]])
        counter = 1
        while counter <= n:
            if counter == i:
                Cx1 = np.kron(Cx1, P0)
                Cx2 = np.kron(Cx2, P1)
            elif counter == j:
                Cx1 = np.kron(Cx1, I)
                Cx2 = np.kron(Cx2, X)
            else:
                Cx1 = np.kron(Cx1, I)
                Cx2 = np.kron(Cx2, I)
            counter += 1
        self.Cx = Cx1 + Cx2


class ParameterizedGate:
    def __init__(self, i, g, n):
        '''
        :param i: act on ith qubit
        :param g: type of pauli matrices(X, Y, Z)
        :param n: qubits quantity
        '''
        self.para = np.random.uniform(-1, 1) * np.pi # initialized uniformly at random in [ − π, + π]
        self.act_bit = i
        self.Gate = np.array([[1.0]]) # Pauli tensor product
        counter = 1
        while counter <= n:
            if counter == i:
                self.Gate = np.kron(self.Gate, g)
            else:
                self.Gate = np.kron(self.Gate, I)
            counter += 1
        self.PGate = lin.expm(-0.5j * self.para * self.Gate)

    '''change parameter'''
    def change_para(self, p):
        self.para = p
        self.PGate = lin.expm(-0.5j * self.para * self.Gate)


def Rzyz(i, n):
    gate_list = []
    gate_list.append(ParameterizedGate(i, Z, n))
    gate_list.append(ParameterizedGate(i, Y, n))
    gate_list.append(ParameterizedGate(i, Z, n))
    return gate_list



'''arbitrary 2-qubits gate, 15 parameterized gate and 3 CNOT gate'''
def two_qubit_gate(i, j, n):
    tq_gate_list = []
    tq_gate_list.extend(Rzyz(i, n))
    tq_gate_list.extend(Rzyz(j, n))
    tq_gate_list.append(CNOT(j, i, n))
    tq_gate_list.append(ParameterizedGate(i, Z, n))
    tq_gate_list.append(ParameterizedGate(j, Y, n))
    tq_gate_list.append(CNOT(i, j, n))
    tq_gate_list.append(ParameterizedGate(j, Y, n))
    tq_gate_list.append(CNOT(j, i, n))
    tq_gate_list.extend(Rzyz(i, n))
    tq_gate_list.extend(Rzyz(j, n))
    return tq_gate_list


'''Layer constituted by 2-qubits gate'''
def layer(n):
    counter = 1
    gate_list = []
    while counter < n:
        gate_list.extend(two_qubit_gate(counter, counter+1, n))
        counter += 2
    counter = 2
    while counter < n:
        gate_list.extend(two_qubit_gate(counter, counter+1, n))
        counter += 2
    return gate_list


def nlayer(n, depth):
    gate_list = []
    for i in range(0, depth):
        gate_list.extend(layer(n))
    return gate_list


def get_E0(i, n):
    E0 = np.array([[1.0]])
    for j in range(1, n + 1):
        if j != i:
            E0 = np.kron(E0, I)
        else:
            E0 = np.kron(E0, P0)
    return E0

def get_Pauli_gate(i, n, G):
    gate = np.array([[1.0]])
    counter = 1
    while counter <= n:
        if counter == i:
            gate = np.kron(gate, G)
        else:
            gate = np.kron(gate, I)
        counter += 1
    return gate

def getX(i, n):
    return get_Pauli_gate(i, n, X)

def getY(i, n):
    return get_Pauli_gate(i, n, Y)

def getZ(i, n):
    return get_Pauli_gate(i, n, Z)

