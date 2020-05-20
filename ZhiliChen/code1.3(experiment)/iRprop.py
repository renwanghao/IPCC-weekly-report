import numpy as np
from scipy import linalg as lin
from matplotlib import pyplot as plt
from Quantum_Circuit import AdversarialQCircuit as QC

dec = 0.5
inc = 1.2
MaxStep = 6e-3 * np.pi
MinStep = 1e-6 * np.pi
InitStep = 1.5e-3 * np.pi


opposite = lambda list: [-i for i in list] # return opposite list


'''iRprop-'''
def gradient_descent(der1, der2, step, para):
    for i in range(0, len(der1)):
        if der1[i] * der2[i] > 0:
            step[i] = min(inc*step[i], MaxStep)
        elif der1[i] * der2[i] < 0:
            step[i] = max(dec*step[i], MinStep)
            der2[i] = 0
        para[i] = para[i] - np.sign(der2[i]) * step[i]


def fidelity(qcircuit):
    gen_state = qcircuit.get_gen_state()
    tar_state = qcircuit.get_tar_state()
    a = np.matmul(gen_state.T.conj(), tar_state)
    f = a * a.conj()
    return f[0][0].real


def trace_distance(qcircuit):
    gen_state = qcircuit.get_gen_dm()
    tar_state = qcircuit.get_tar_dm()
    a = lin.sqrtm(np.matmul((tar_state - gen_state), (tar_state - gen_state)))
    td = np.trace(a) * 0.5
    return td.real


def loss(qcircuit):
    trace1 = qcircuit.observet()
    trace2 = qcircuit.observeg_without_noise()
    return (0.5 * trace1.real - 0.5 * trace2.real)


def iteration(qcircuit, m):
    y1 = []
    y2 = []
    y3 = []
    gen_para = qcircuit.get_generator_para()
    gen_der1 = [0] * len(gen_para)
    gen_step = [InitStep] * len(gen_para)
    dis_para = qcircuit.get_discriminator_para()
    dis_der1 = [0] * len(dis_para)
    dis_step = [InitStep] * len(dis_para)
    # execute m iterations, each iteration trains generator once and trains discriminator 5 times
    for i in range(0, m):
        gen_der2 = qcircuit.get_generator_der_list()
        gradient_descent(gen_der1, gen_der2, gen_step, gen_para)
        qcircuit.change_generator_para(gen_para)
        gen_der1 = gen_der2[:]
        y2.append(loss(qcircuit))
        for i in range(0, 5):
            dis_der2 = opposite(qcircuit.get_discriminator_der_list())
            gradient_descent(dis_der1, dis_der2, dis_step, dis_para)
            qcircuit.change_discriminator_para(dis_para)
            dis_der1 = dis_der2[:]
        y3.append(loss(qcircuit))
        # print(fidelity(qcircuit))
        y1.append(fidelity(qcircuit))
    return y1, y2, y3
    # x = np.arange(0, m, 1)
    # plt.xlabel("iteration")
    # plt.ylabel("fidelity")
    # plt.plot(x, y1, 'd')
    # plt.savefig('./fidelity.png')
    # plt.show()

# cir = QC.QCircuitSimulator(1, 1, 1, 2, "Bit_Flip")
# iteration(cir, 200)

# def simulate():
#     cir = QC.QCircuitSimulator(1, 1, 1, 2, "Depolarizing_Channel")
#     # noise_name = Bit_Flip, Phase_Flip, Bit_Phase_Flip, Depolarizing_Channel,
#     iteration(cir)
#     a = cir.get_tar_state()
#     print(a)
#     b = cir.get_gen_state()
#     print(b)
#
#
# simulate()
# c = QC.QCircuitSimulator(1,1,1,2)
# a = c.get_generator_der_list()


'''loss function'''
# def p_error(qcircuit):
#     qcircuit.generate_state()
#     qcircuit.discriminate_state()
#     ancqubitstate = np.matmul(qcircuit.register.state, qcircuit.register.state.T.conj())
#     ancprob0 = np.trace(np.matmul(qcircuit.E0, ancqubitstate))
#     qcircuit.register_initial()
#     qcircuit.target_state()
#     qcircuit.discriminate_state()
#     ancqubitstate = np.matmul(qcircuit.register.state, qcircuit.register.state.T.conj())
#     ancprob1 = np.trace(np.matmul(qcircuit.E0, ancqubitstate))
#     qcircuit.register_initial()
#     perr = ancprob1 * 0.5 - ancprob0 * 0.5
#     return perr.real


# def min_generator(qcircuit):
#     gen_para = qcircuit.get_generator_para()
#     gen_der1 = [0] * len(gen_para)
#     gen_step = [InitStep] * len(gen_para)
#     # while True:
#     #     curperr = p_error(qcircuit)
#     for i in range(0, 2):
#         gen_der2 = qcircuit.get_generator_der_list()
#         gradient_descent(gen_der1, gen_der2, gen_step, gen_para)
#         qcircuit.change_generator_para(gen_para)
#         gen_der1 = gen_der2[:]
#         # if abs(curperr - p_error(qcircuit)) < 0.001:
#         #     break
#         print(p_error(qcircuit))
#
#
# def max_discriminator(qcircuit):
#     dis_para = qcircuit.get_discriminator_para()
#     dis_der1 = [0] * len(dis_para)
#     dis_step = [InitStep] * len(dis_para)
#     # while True:
#     #     curperr = p_error(qcircuit)
#     for i in range(0, 10):
#         dis_der2 = qcircuit.get_discriminator_der_list()
#         gradient_ascent(dis_der1, dis_der2, dis_step, dis_para)
#         qcircuit.change_discriminator_para(dis_para)
#         dis_der1 = dis_der2[:]
#         # if abs(curperr - p_error(qcircuit)) < 0.001:
#         #     break
#         print(p_error(qcircuit))






