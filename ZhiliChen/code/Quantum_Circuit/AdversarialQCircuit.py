import Quantum_Circuit.QGate as QG
import Quantum_Circuit.QRegister as QR
import numpy as np


class QCircuitSimulator:

    def __init__(self, TDepth, GDepth, DDepth, n):
        '''TDepth: Depth of target state circuit
        GDepth: Depth of generator circuit
        DDepth: Depth of discriminator circuit
        n: qubits digit of target state
        '''
        self.qnum = n + 1 # qbum is qubits digit of whole circuit, should add to digit of ancillary qubit
        self.d = 2 ** (n + 1) # dimension of hilbert space
        self.register = QR.QRegister(n + 1) # register used to save state of the whole qubits
        self.main_register = QR.QRegister(n) # register used to save generator state, used for calculate fidelity and trace distance
        self.anc_register = QR.QRegister(1) # register used to save ancillary qubit state, state is always ([[1.0],[0.0]]), used for kron
        self.E0 = np.array([[1.0]]) # projector
        self.target_list = [] # list of target circuit
        self.ggate_list = [] # list of generate circuit
        self.dgate_list = [] # list of discriminate circuit
        # get projector E0
        for i in range(1, n+2):
            if i != n+1:
                self.E0 = np.kron(self.E0, QG.I)
            else:
                self.E0 = np.kron(self.E0, QG.P0)
        # create layer instances(count is equal to TDepth), extend list by these layer instance
        counter = 1
        while counter <= TDepth:
            self.target_list.extend(QG.Layer(n).gate_list)
            counter += 1
        # get list of generate circuit
        counter = 1
        while counter <= GDepth:
            self.ggate_list.extend(QG.Layer(n).gate_list)
            counter += 1
        # get list of discriminate circuit
        counter = 1
        while counter <= DDepth:
            self.dgate_list.extend(QG.Layer(n+1).gate_list)
            counter += 1

    '''initial state of entire registers'''
    def register_initial(self):
        self.register.initial()
        self.main_register.initial()
        self.anc_register.initial()

    '''get target state'''
    def target_state(self):
        tcircuit = np.eye(int(self.d/2))
        for i in self.target_list:
            if isinstance(i, QG.CNOT):
                tcircuit = np.dot(i.Cx, tcircuit)
            else:
                tcircuit = np.dot(i.PGate, tcircuit)
        self.main_register.state = np.dot(tcircuit, self.main_register.state)
        self.register.state = np.kron(self.main_register.state, self.anc_register.state) # kron with ancillary qubit state to get state of entire qubits

    '''get generate state, generator act on the main_register state'''
    def generate_state(self):
        gcircuit = np.eye(int(self.d/2))
        for i in self.ggate_list:
            if isinstance(i, QG.CNOT):
                gcircuit = np.matmul(i.Cx, gcircuit)
            else:
                gcircuit = np.matmul(i.PGate, gcircuit)
        self.main_register.state = np.matmul(gcircuit, self.main_register.state)
        self.register.state = np.kron(self.main_register.state, self.anc_register.state)

    '''use density matrix to get state after generating'''
    # def generate_state1(self):
    #     gcircuit = np.eye(2 ** (self.qnum-1))
    #     for i in self.ggate_list:
    #         if isinstance(i, QG.CNOT):
    #             gcircuit = np.matmul(i.Cx, gcircuit)
    #         else:
    #             gcircuit = np.matmul(i.PGate, gcircuit)
    #     self.main_register.state1 = np.matmul(np.matmul(gcircuit, self.main_register.state1),gcircuit.T.conj())
    #     self.register.state1 = np.kron(self.main_register.state1, self.anc_register.state1)

    '''get state after discriminating, discriminator act on the register state'''
    def discriminate_state(self):
        dcircuit = np.eye(self.d)
        for i in self.dgate_list:
            if isinstance(i, QG.CNOT):
                dcircuit = np.matmul(i.Cx, dcircuit)
            else:
                dcircuit = np.matmul(i.PGate, dcircuit)
        self.register.state = np.matmul(dcircuit, self.register.state)

    '''use density matrix to get state after discriminating'''
    # def discriminate_state1(self):
    #     dcircuit = np.eye(self.d)
    #     for i in self.dgate_list:
    #         if isinstance(i, QG.CNOT):
    #             dcircuit = np.matmul(i.Cx, dcircuit)
    #         else:
    #             dcircuit = np.matmul(i.PGate, dcircuit)
    #     self.register.state1 = np.matmul(np.matmul(dcircuit, self.register.state1), dcircuit.T.conj())

    '''change phase to get derivative'''
    def generator_derivative(self, i):
        p = self.ggate_list[i].para
        self.ggate_list[i].change_para(np.pi/2+p) # change phase to np.pi/2+p
        self.generate_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj()) # get density matrix
        ancprob0 = np.trace(np.matmul(self.E0, ancqubitstate)) # trace(probability of ancillary qubit state) of part1
        # count = 0
        # for j in range(0,100):
        #     if(np.random.rand() < ancprob0):
        #         count += 1
        # part1 = count / 100
        self.register_initial() # register state should initial to 0 state
        self.ggate_list[i].change_para(p-np.pi/2)
        self.generate_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj())
        ancprob1 = np.trace(np.matmul(self.E0, ancqubitstate)) # trace of part2
        # count = 0
        # for j in range(0, 100):
        #     if (np.random.rand() < ancprob1):
        #         count += 1
        # part2 = count / 100
        self.register_initial()
        self.ggate_list[i].change_para(p) # change phase to original p
        der = -0.5 * 0.5 * (ancprob0 - ancprob1).real # trace1 substract trace2 getting derivative
        return der

    '''use analytic expression to get derivative'''
    # def generator_derivative1(self, i):
    #     gcircuit1 = np.eye(2 ** (self.qnum-1))
    #     for j in range(0, i+1):
    #         if isinstance(self.ggate_list[j], QG.CNOT):
    #             gcircuit1 = np.matmul(self.ggate_list[j].Cx, gcircuit1)
    #         else:
    #             gcircuit1 = np.matmul(self.ggate_list[j].PGate, gcircuit1)
    #     h = self.ggate_list[i].Gate
    #     self.main_register.state1 = np.matmul(gcircuit1, np.matmul(self.main_register.state1, gcircuit1.T.conj()))
    #     self.main_register.state1 = np.matmul(h, self.main_register.state1) - np.matmul(self.main_register.state1, h)
    #     gcircuit2 = np.eye(2 ** (self.qnum - 1))
    #     for j in range(i+1, len(self.ggate_list)):
    #         if isinstance(self.ggate_list[j], QG.CNOT):
    #             gcircuit2 = np.matmul(self.ggate_list[j].Cx, gcircuit2)
    #         else:
    #             gcircuit2 = np.matmul(self.ggate_list[j].PGate, gcircuit2)
    #     self.main_register.state1 = np.matmul(gcircuit2, np.matmul(self.main_register.state1, gcircuit2.T.conj()))
    #     self.register.state1 = np.kron(self.main_register.state1, self.anc_register.state1)
    #     dcircuit = np.eye(self.d)
    #     for i in self.dgate_list:
    #         if isinstance(i, QG.CNOT):
    #             dcircuit = np.matmul(i.Cx, dcircuit)
    #         else:
    #             dcircuit = np.matmul(i.PGate, dcircuit)
    #     self.register.state1 = np.matmul(np.matmul(dcircuit, self.register.state1), dcircuit.T.conj())
    #     der = 0.25j * np.trace(np.matmul(self.E0, self.register.state1))
    #     self.register_initial()
    #     return der.real

    def discriminator_derivative(self, i):
        p = self.dgate_list[i].para
        self.dgate_list[i].change_para(np.pi/2+p)
        self.generate_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj())
        ancprob0 = np.trace(np.matmul(self.E0, ancqubitstate))
        # count = 0
        # for j in range(0, 100):
        #     if (np.random.rand() < ancprob0):
        #         count += 1
        # part1 = count / 100
        self.register_initial()
        self.dgate_list[i].change_para(p-np.pi/2)
        self.generate_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj())
        ancprob1 = np.trace(np.matmul(self.E0, ancqubitstate))
        # count = 0
        # for j in range(0, 100):
        #     if (np.random.rand() < ancprob1):
        #         count += 1
        # part2 = count / 100
        self.register_initial()
        der1 = -0.5 * 0.5 * (ancprob0 - ancprob1).real # derivative of part1
        self.dgate_list[i].change_para(np.pi/2+p)
        self.target_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj())
        ancprob0 = np.trace(np.matmul(self.E0, ancqubitstate))
        # count = 0
        # for j in range(0, 100):
        #     if (np.random.rand() < ancprob0):
        #         count += 1
        # part1 = count / 100
        self.register_initial()
        self.dgate_list[i].change_para(p-np.pi/2)
        self.target_state()
        self.discriminate_state()
        ancqubitstate = np.matmul(self.register.state, self.register.state.T.conj())
        ancprob1 = np.trace(np.matmul(self.E0, ancqubitstate))
        # count = 0
        # for j in range(0, 100):
        #     if (np.random.rand() < ancprob1):
        #         count += 1
        # part2 = count / 100
        self.register_initial()
        self.dgate_list[i].change_para(p)
        der2 = 0.5 * 0.5 * (ancprob0 - ancprob1).real # derivative of part2
        der = der1 + der2
        return der

    '''return list of discriminator parameters'''
    def get_discriminator_para(self):
        list = []
        for i in self.dgate_list:
            if isinstance(i, QG.ParameterizedGate):
                list.append(i.para)
        return list

    '''return list of generator parameters'''
    def get_generator_para(self):
        list = []
        for i in self.ggate_list:
            if isinstance(i, QG.ParameterizedGate):
                list.append(i.para)
        return list

    '''return list of discriminator derivative'''
    def get_discriminator_der_list(self):
        list = []
        for i in range(0, len(self.dgate_list)):
            if isinstance(self.dgate_list[i], QG.ParameterizedGate):
                a = self.discriminator_derivative(i)
                if abs(a) < 1e-15: # eliminate error
                    a = 0.0
                list.append(a)
        return list

    '''return list of generator derivative'''
    def get_generator_der_list(self):
        list = []
        for i in range(0, len(self.ggate_list)):
            if isinstance(self.ggate_list[i], QG.ParameterizedGate):
                a = self.generator_derivative(i)
                if abs(a) < 1e-15:
                    a = 0.0
                list.append(a)
        return list

    '''use analytic expression to get list of derivative'''
    # def get_generator_der_list1(self):
    #     list = []
    #     for i in range(0, len(self.ggate_list)):
    #         if isinstance(self.ggate_list[i], QG.ParameterizedGate):
    #             list.append(self.generator_derivative1(i))
    #     return list

    '''change entire parameters of generator list'''
    def change_generator_para(self, paralist):
        j = 0
        for i in self.ggate_list:
            if isinstance(i, QG.ParameterizedGate):
                i.change_para(paralist[j])
                j += 1

    '''change entire parameters of discriminator list'''
    def change_discriminator_para(self, paralist):
        j = 0
        for i in self.dgate_list:
            if isinstance(i, QG.ParameterizedGate):
                i.change_para(paralist[j])
                j += 1

