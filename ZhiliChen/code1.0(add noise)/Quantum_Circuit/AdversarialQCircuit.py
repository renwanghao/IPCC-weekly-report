import Quantum_Circuit.QGate as QG
import Quantum_Circuit.QRegister as QR
import numpy as np


Pfnoise = 0.4
Pdnoise = 0.4


elim_err = lambda num: 0.0 if abs(num) < 10e-15 else num # eliminate error


'''change gate para by paralist'''
def change_para(gatelist, paralist):
    j = 0
    for i in gatelist:
        if isinstance(i, QG.ParameterizedGate):
            i.change_para(paralist[j])
            j += 1


'''return paralist of gatelist'''
def get_para_list(gatelist):
    list = []
    for i in gatelist:
        if isinstance(i, QG.ParameterizedGate):
            list.append(i.para)
    return list


'''return product of gatelist'''
def circuit_product(gatelist):
    d = len(gatelist[0].Gate)
    circuit = np.eye(d)
    for i in gatelist:
        if isinstance(i, QG.CNOT):
            circuit = np.dot(i.Cx, circuit)
        else:
            circuit = np.dot(i.PGate, circuit)
    return circuit


class QCircuitSimulator:
    def __init__(self, TDepth, GDepth, DDepth, n):
        '''TDepth: Depth of target state circuit
        GDepth: Depth of generator circuit
        DDepth: Depth of discriminator circuit
        n: qubits digit of target state
        '''
        self.qnum = n + 1 # qbum is qubits digit of whole circuit, should add to digit of ancillary qubit
        self.d = 2 ** (n + 1) # dimension of hilbert space
        self.register = QR.QRegister(n, 1) # register used to save state of the whole qubits
        # get projector E0
        self.E0 = QG.get_E0(n+1, n+1)
        # create layer instances(count is equal to TDepth), extend list by these layer instance
        self.target_list = QG.nlayer(n, TDepth)
        # get list of generate circuit
        self.ggate_list = QG.nlayer(n, GDepth)
        # get list of discriminate circuit
        self.dgate_list = QG.nlayer(n+1, DDepth)

    '''initial state of entire registers'''
    def register_initial(self):
        self.register.initial()

    '''get target state'''
    def target_state(self):
        tcircuit = circuit_product(self.target_list)
        self.register.change_main_state(tcircuit)

    '''get generate state, generator act on the main_register state'''
    def generate_state(self):
        gcircuit = circuit_product(self.ggate_list)
        self.register.change_main_state(gcircuit)

    '''get state after discriminating, discriminator act on the register state'''
    def discriminate_state(self):
        dcircuit = circuit_product(self.dgate_list)
        self.register.change_state(dcircuit)


    '''return generate state'''
    def get_gen_state(self):
        self.generate_state()
        state = self.register.get_main_state()
        self.register_initial()
        return state

    '''return generate density matrix'''
    def get_gen_dm(self):
        self.generate_state()
        density_mat = self.register.get_main_density()
        self.register_initial()
        return density_mat

    '''return target state'''
    def get_tar_state(self):
        self.target_state()
        state = self.register.get_main_state()
        self.register_initial()
        return state

    '''return target density matrix'''
    def get_tar_dm(self):
        self.target_state()
        density_mat = self.register.get_main_density()
        self.register_initial()
        return density_mat

    def observe(self):
        self.discriminate_state()
        #self.flip_noise(self.qnum)
        #self.depolarize_noise(self.qnum)
        density_mat = self.register.density
        ancprob = np.trace(np.matmul(self.E0, density_mat))
        self.register.initial()  # register state should initial to 0 state
        return ancprob

    '''observe generate state and get expectation'''
    def observeg(self):
        self.generate_state()
        return self.observe()

    '''observe target state and get expectation'''
    def observet(self):
        self.target_state()
        return self.observe()

    '''change phase to get derivative, return derivative of ith generate gate'''
    def generator_derivative(self, i):
        p = self.ggate_list[i].para
        self.ggate_list[i].change_para(np.pi/2+p) # change phase to np.pi/2+p
        ancprob0 = self.observeg() # trace(probability of ancillary qubit state) of part1
        self.ggate_list[i].change_para(p-np.pi/2)
        ancprob1 = self.observeg() # trace of part2
        self.ggate_list[i].change_para(p) # change phase to original p
        der = -0.5 * 0.5 * (ancprob0 - ancprob1).real # trace1 substract trace2 getting derivative
        return der

    '''return derivative of ith discrimnate gate'''
    def discriminator_derivative(self, i):
        p = self.dgate_list[i].para
        self.dgate_list[i].change_para(np.pi/2+p)
        ancprob0 = self.observeg()
        ancprob2 = self.observet()
        self.dgate_list[i].change_para(p-np.pi/2)
        ancprob1 = self.observeg()
        ancprob3 = self.observet()
        self.dgate_list[i].change_para(p)
        der1 = -0.5 * 0.5 * (ancprob0 - ancprob1).real # derivative of part1
        der2 = 0.5 * 0.5 * (ancprob2 - ancprob3).real # derivative of part2
        der = der1 + der2
        return der

    '''return list of discriminator parameters'''
    def get_discriminator_para(self):
        return get_para_list(self.dgate_list)

    '''return list of generator parameters'''
    def get_generator_para(self):
        return get_para_list(self.ggate_list)

    '''return list of discriminator derivative'''
    def get_discriminator_der_list(self):
        list = []
        for i in range(0, len(self.dgate_list)):
            if isinstance(self.dgate_list[i], QG.ParameterizedGate):
                list.append(elim_err(self.discriminator_derivative(i)))
        return list

    '''return list of generator derivative'''
    def get_generator_der_list(self):
        list = []
        for i in range(0, len(self.ggate_list)):
            if isinstance(self.ggate_list[i], QG.ParameterizedGate):
                list.append(elim_err(self.generator_derivative(i)))
        return list

    '''change parameters of generator list'''
    def change_generator_para(self, paralist):
        change_para(self.ggate_list, paralist)

    '''change parameters of discriminator list'''
    def change_discriminator_para(self, paralist):
        change_para(self.dgate_list, paralist)

    def flip_noise(self, n):
        dmat = self.register.density
        for i in range(1, n + 1):
            x = QG.getX(i, self.qnum)
            dmat = Pfnoise * np.matmul(x, np.matmul(dmat, x)) + (1 - Pfnoise) * dmat
        self.register.density = dmat

    def depolarize_noise(self, n):
        dmat = self.register.density
        for i in range(1, n + 1):
            x = QG.getX(i, self.qnum)
            y = QG.getY(i, self.qnum)
            z = QG.getZ(i, self.qnum)
            dmat = (1 - 0.75 * Pdnoise) * dmat + (Pdnoise / 4) * (np.matmul(x, np.matmul(dmat, x)) + np.matmul(y, np.matmul(dmat, y)) + np.matmul(z, np.matmul(dmat, z)))
        self.register.density = dmat

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

    '''use analytic expression to get list of derivative'''
    # def get_generator_der_list1(self):
    #     list = []
    #     for i in range(0, len(self.ggate_list)):
    #         if isinstance(self.ggate_list[i], QG.ParameterizedGate):
    #             list.append(self.generator_derivative1(i))
    #     return list