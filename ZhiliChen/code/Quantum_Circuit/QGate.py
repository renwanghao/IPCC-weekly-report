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


'''arbitrary 2-qubits gate, 15 parameterized gate and 3 CNOT gate'''
class TwoQbitGate:
    def __init__(self, i, j, n):
        self.tq_gate_list = []
        self.tq_gate_list.append(ParameterizedGate(i, Z, n))
        self.tq_gate_list.append(ParameterizedGate(i, Y, n))
        self.tq_gate_list.append(ParameterizedGate(i, Z, n))
        self.tq_gate_list.append(ParameterizedGate(j, Z, n))
        self.tq_gate_list.append(ParameterizedGate(j, Y, n))
        self.tq_gate_list.append(ParameterizedGate(j, Z, n))
        self.tq_gate_list.append(CNOT(j, i, n))
        self.tq_gate_list.append(ParameterizedGate(i, Z, n))
        self.tq_gate_list.append(ParameterizedGate(j, Y, n))
        self.tq_gate_list.append(CNOT(i, j, n))
        self.tq_gate_list.append(ParameterizedGate(j, Y, n))
        self.tq_gate_list.append(CNOT(j, i, n))
        self.tq_gate_list.append(ParameterizedGate(i, Z, n))
        self.tq_gate_list.append(ParameterizedGate(i, Y, n))
        self.tq_gate_list.append(ParameterizedGate(i, Z, n))
        self.tq_gate_list.append(ParameterizedGate(j, Z, n))
        self.tq_gate_list.append(ParameterizedGate(j, Y, n))
        self.tq_gate_list.append(ParameterizedGate(j, Z, n))


'''Layer constituted by 2-qubits gate'''
class Layer:
    def __init__(self, n):
        counter = 1
        self.gate_list = []
        while counter < n:
            self.gate_list.extend(TwoQbitGate(counter, counter+1, n).tq_gate_list)
            counter += 2
        counter = 2
        while counter < n:
            self.gate_list.extend(TwoQbitGate(counter, counter+1, n).tq_gate_list)
            counter += 2






