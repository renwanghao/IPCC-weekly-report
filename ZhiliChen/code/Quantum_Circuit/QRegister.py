import numpy as np


Zero = np.array([[1.0],[0.0]])
One = np.array([[0.0],[1.0]])


class QRegister:
    '''initial state is 0 state'''
    def __init__(self, n):
        self.length = n
        counter = 1
        self.state = np.array([[1.0]])
        while counter <= n:
            self.state = np.kron(self.state, Zero)
            counter += 1
        # self.state1 = np.matmul(self.state, self.state.T)

    '''initial to 0 state'''
    def initial(self):
        counter = 1
        self.state = np.array([[1.0]])
        while counter <= self.length:
            self.state = np.kron(self.state, Zero)
            counter += 1
        # self.state1 = np.matmul(self.state, self.state.T)


    # def simulate(self, pslist):
    #     counter = 0
    #     self.state = np.array([[1.0]])
    #     while counter < self.length:
    #         singlebitstate = pslist[counter][0] * Zero + pslist[counter][1] * One
    #         self.state = np.kron(self.state, singlebitstate)
    #         counter += 1
