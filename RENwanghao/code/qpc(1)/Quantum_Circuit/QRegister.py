import numpy as np


Zero = np.array([[1.0], [0.0]])
One = np.array([[0.0], [1.0]])


def get_initial_state(n):
    state = np.array([[1.0]])
    for i in range(0, n):
        state = np.kron(state, Zero)
    return state


class QRegister1:
    def __init__(self, n):
        self.length = n
        self.state = get_initial_state(n)
        self.density = np.matmul(self.state, self.state.T) # density matrix

    def initial(self):
        self.state = get_initial_state(self.length)
        self.density = np.matmul(self.state, self.state.T)

    def change_state(self, circuit):
        self.state = np.matmul(circuit, self.state)
        self.density = np.matmul(self.state, self.state.T.conj())


class QRegister:
    '''initial state is 0 state'''
    def __init__(self, n, m):
        self.length = n + m
        self.main_register = QRegister1(n) # register used to save generator state, used for calculate fidelity and trace distance
        self.anc_register = QRegister1(m) # register used to save ancillary qubit state, state is always ([[1.0],[0.0]]), used for kron
        self.state = get_initial_state(m + n)
        self.density = np.matmul(self.state, self.state.T)

    def get_main_state(self):
        return self.main_register.state

    def get_main_density(self):
        return self.main_register.density

    def change_state(self, circuit):
        self.state = np.matmul(circuit, self.state)
        self.density = np.matmul(self.state, self.state.T.conj())

    def change_main_state(self, circuit):
        self.main_register.change_state(circuit)
        self.state = np.kron(self.main_register.state, self.anc_register.state) # kron with ancillary qubit state to get state of entire qubits
        self.density = np.matmul(self.state, self.state.T.conj())

    '''initial to 0 state'''
    def initial(self):
        self.main_register.initial()
        self.anc_register.initial()
        self.state = get_initial_state(self.length)
        self.density = np.matmul(self.state, self.state.T)
