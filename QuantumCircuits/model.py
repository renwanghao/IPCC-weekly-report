import numpy as np
from QuantumCircuits.quantumcircuits import Quantum_Circuit, zero_ListTheta, random_ListTheta
from QuantumCircuits.quantumcircuits import Identity, P0, P1
size = 2


def get_zero_state(size):
    '''
        get the zero quantum state |0,...0>
        形式为矩阵，
    :param size:
    :return:
    '''
    zero_state = np.zeros(2 ** size)
    zero_state[0] = 1
    zero_state = np.asmatrix(zero_state).T
    return zero_state


def get_p0():
    zero_state = np.zeros(2)
    zero_state[0] = 1
    zero_state = np.asmatrix(zero_state).T
    zero_stateT = zero_state.T
    p0 = np.matmul(zero_state, zero_stateT)
    p0 = np.asmatrix(p0)
    return p0


def get_p0_size(size):
    p0 = get_p0()
    for i in range(size - 1):
        p0 = np.kron(p0, p0)
    return p0
def get_p0_size_dis(size):
    p0 = get_p0()
    for i in range(size - 2):
        p0 = np.kron(p0, p0)
    return p0

def get_real_state(size):
    # real_state = np.zeros(2 ** size)
    # real_state = []
    # for i in range(2 ** size):
    #     a = np.random.rand()
    #     b = 1 - a*a
    #     c = complex(a, b)
    #     real_state.append(c)
    # # print(c)
    # # print(real_state)
    # mat = np.asmatrix(real_state)
    # mat = mat.getH()
    # print(mat)  # 给出一个8*1 的复数矩阵

    # 上面的方法不对。下面先用生成器随便生成一个态作为真实的态。

    gen_theta = random_ListTheta(15)  # 生成theta的随机参数list
    gen = Generator(size, gen_theta)  # define gen，输入size和参数
    mat_gen, gen_theta = gen.qc.Vtheta1()
    zero_state = get_zero_state(size)
    mat = np.matmul(mat_gen, zero_state)
    return mat


def compute_cost(gen_mat, gen_theta, dis_mat, dis_phi, real_state):
    '''
    :param mat_gen: G
    :param gen_theta:
    :param mat_dis: D
    :param dis_phi:
    :param real_state: psi
    :return:
    '''
    G = gen_mat
    theta = gen_theta

    D = dis_mat  # 判别器的矩阵表示
    phi = dis_phi  # 判别器的参数

    zero_state = get_zero_state(size)
    # fake_state = np.matmul(G, zero_state)
    psi = real_state  # size * 1 的矩阵

    E0 = np.kron(Identity(size), P0)

    theta_plus = []
    theta_mins = []
    new_grad = []
    # =list()
    # 如果要表示相同的列表，第一个当中的元素必须用逗号隔开，则第二种不需要用逗号隔开，这就是a=[]和a=list()的区别
    # a = [1, 2, 3] 输出[1, 2, 3]      a=list("123") 输出['1', '2', '3']
    Pt = 1 / 2
    Pg = 1 / 2
    p0 = get_p0()  # 2*2 matrix |0><0|
    p0n = get_p0_size(size)  # p0 tensor n 次，n=size
    tem1 = np.kron(
        (np.matmul(psi, psi.getH())), p0)
    tem2 = np.kron(
        (np.matmul(
            G,
            np.matmul(
                p0n,
                np.asmatrix(G).getH()))),
        p0)
    V_theta_psi1 = np.trace(
        np.matmul(
            E0,
            np.matmul(
                D,
                np.matmul(
                    tem1,
                    np.asmatrix(D).getH()))))
    V_theta_psi2 = np.trace(
        np.matmul(
            E0,
            np.matmul(
                D,
                np.matmul(
                    tem2,
                    np.asmatrix(D).getH()))))
    V_theta_psi = Pt * V_theta_psi1 - Pg * V_theta_psi2

    return V_theta_psi


def compute_fidelity(mat_gen, state, real_state):
    fake_state = np.dot(mat_gen, state)
    fidelity = np.abs(
        np.asscalar(
            np.matmul(
                real_state.getH(),
                fake_state))) ** 2
    # real_state and fake_state is a matrix in (2**size) * 1.
    # np.matmul 得到最后的形式是矩阵，但是我们需要的是标量。用np.asscalar
    # 两个态都是纯态，那么保真度是他们内积的平方
    # https://en.wikipedia.org/wiki/Fidelity_of_quantum_states#Definition
    if fidelity < 0:
        print('error compute_fidelity < 0 ')
    elif fidelity > 1:
        print('error compute_fidelity > 0')
    return fidelity


class Generator:
    '''
        生成器类需要system_size参数,和Theta参数list
    '''

    def __init__(self, system_size, param):
        self.size = system_size
        self.param = param
        self.qc = self.init_qcircuit()

    def init_qcircuit(self):
        '''
        作用是 初始化量子线路
        是 Quantum_Circuit 量子线路类 的 一个实例,他会给量子线路一个size和name,以及参数theta的list。
        返回一个量子线路的实例。
        :return:
        '''
        # lst0 = zero_ListTheta(15)
        qcircuit = Quantum_Circuit(self.size, "generator", self.param)
        return qcircuit

    def grad_theta(self, gen_mat, gen_theta, dis_mat, dis_phi, real_state):

        G = gen_mat
        theta = gen_theta

        D = dis_mat  # 判别器的矩阵表示
        phi = dis_phi  # 判别器的参数

        zero_state = get_zero_state(self.size)
        # fake_state = np.matmul(G, zero_state)

        E0 = np.kron(Identity(self.size), P0)

        theta_plus = []
        theta_mins = []
        new_grad = []
        # =list()
        # 如果要表示相同的列表，第一个当中的元素必须用逗号隔开，则第二种不需要用逗号隔开，这就是a=[]和a=list()的区别
        # a = [1, 2, 3] 输出[1, 2, 3]      a=list("123") 输出['1', '2', '3']
        theta_mins = theta
        theta_plus = theta   # 每次重新赋值是因为只有一项变化，其他不变
        for i in range(len(theta)):
            # theta_mins = theta
            # theta_plus = theta   # 每次重新赋值是因为只有一项变化，其他不变
            theta_plus[i] = theta[i] + np.pi / 2
            theta_mins[i] = theta[i] - np.pi / 2
            # theta_itr = theta[i]
            gen_plus = Generator(self.size, theta_plus)
            gen_mins = Generator(self.size, theta_mins)
            G_plus, theta_plus = gen_plus.qc.Vtheta1()
            G_mins, theta_mins = gen_mins.qc.Vtheta1()
            Pt = 1 / 2
            Pg = 1 / 2
            p0 = get_p0()
            p0n = get_p0_size(self.size)
            tem1 = np.kron(
                (np.matmul(
                    G_plus,
                    np.matmul(
                        p0n,
                        np.asmatrix(G_plus).getH()))),
                p0)
            tem2 = np.kron(
                (np.matmul(
                    G_mins,
                    np.matmul(
                        p0n,
                        np.asmatrix(G_mins).getH()))),
                p0)
            theta_partial_derivatives1 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D, np.matmul(
                            tem1, np.asmatrix(D).getH()))))
            theta_partial_derivatives2 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D, np.matmul(
                            tem2, np.asmatrix(D).getH()))))

            theta_partial_derivatives = -1 / 2 * Pg * \
                (theta_partial_derivatives1 + theta_partial_derivatives2)
            new_grad.append(theta_partial_derivatives)

        return new_grad

    def update_theta(self, gen_mat, gen_theta, dis_mat, dis_phi, real_state):
        new_angle = []
        old_gen_theta = gen_theta  # 原来的参数theta
        epsilon = 0.01
        grad_list = self.grad_theta(
            gen_mat,
            gen_theta,
            dis_mat,
            dis_phi,
            real_state)  # 新的参数theta
        for i in range(len(gen_theta)):
            new_angle.append(old_gen_theta[i] - epsilon * grad_list[i])
        theta = new_angle
        return theta


class Discriminator:

    def __init__(self, system_size, param):
        self.size = system_size + 1  # equal to G's system size +1
        self.param = param  # euqal to (system_size-1) *15
        self.qc = self.init_qcircuit()

    def init_qcircuit(self):

        # lst0 = zero_ListTheta(15)
        qcircuit = Quantum_Circuit(self.size, "generator", self.param)
        return qcircuit

    def grad_phi(self, gen_mat, gen_theta, dis_mat, dis_phi, real_state):

        G = gen_mat
        theta = gen_theta

        D = dis_mat  # 判别器的矩阵表示
        phi = dis_phi  # 判别器的参数

        zero_state = get_zero_state(self.size)
        # fake_state = np.matmul(G, zero_state)
        psi = real_state

        E0 = np.kron(Identity(self.size - 1), P0)

        phi_plus = []
        phi_mins = []
        new_grad = []
        # =list()
        # 如果要表示相同的列表，第一个当中的元素必须用逗号隔开，则第二种不需要用逗号隔开，这就是a=[]和a=list()的区别
        # a = [1, 2, 3] 输出[1, 2, 3]      a=list("123") 输出['1', '2', '3']

        for i in range(len(phi)):
            phi_mins = phi
            phi_plus = phi  # 每次重新赋值是因为只有一项变化，其他不变
            phi_plus[i] = phi[i] + np.pi / 2
            phi_mins[i] = phi[i] - np.pi / 2
            # theta_itr = theta[i]
            dis_plus = Discriminator(self.size, phi_plus)
            dis_mins = Discriminator(self.size, phi_mins)
            D_plus, phi_plus = dis_plus.qc.Vtheta2()
            D_mins, phi_mins = dis_mins.qc.Vtheta2()
            Pt = 1 / 2
            Pg = 1 / 2
            p0 = get_p0()
            p0n = get_p0_size_dis(self.size)
            tem1 = np.kron(
                (np.matmul(psi, psi.getH())), p0)
            tem2 = np.kron(
                (np.matmul(
                    G,
                    np.matmul(
                        p0n,
                        np.asmatrix(G).getH()))),
                p0)
            phi_partial_derivatives1 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D_plus, np.matmul(
                            tem1, np.asmatrix(D_plus).getH()))))
            phi_partial_derivatives2 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D_mins, np.matmul(
                            tem1, np.asmatrix(D_mins).getH()))))
            phi_partial_derivatives3 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D_plus, np.matmul(
                            tem2, np.asmatrix(D_plus).getH()))))
            phi_partial_derivatives4 = np.trace(
                np.matmul(
                    E0, np.matmul(
                        D_mins, np.matmul(
                            tem2, np.asmatrix(D_mins).getH()))))
            phi_partial_derivatives = 1 / 2 * Pt * (
                phi_partial_derivatives1 - phi_partial_derivatives2) - 1 / 2 * Pg * (
                phi_partial_derivatives3 - phi_partial_derivatives4)
            new_grad.append(phi_partial_derivatives)

        return new_grad

    def update_phi(self, gen_mat, gen_theta, dis_mat, dis_phi, real_state):
        new_angle = []
        old_gen_phi = dis_phi  # 原来的参数theta
        eta = 0.01
        grad_list = self.grad_phi(
            gen_mat,
            gen_theta,
            dis_mat,
            dis_phi,
            real_state)  # 新的参数theta
        for i in range(len(dis_phi)):
            new_angle.append(old_gen_phi[i] + eta * grad_list[i])
        phi = new_angle
        return phi
