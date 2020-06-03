
import numpy as np
from scipy import linalg as lin
from matplotlib import pyplot as plt
from Quantum_Circuit import AdversarialQCircuit as QC

import cv2

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
    # b = np.abs(np.asscalar(np.matmul(gen_state.T.conj(), tar_state))) ** 2
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
    i = 0
    while fidelity(qcircuit) < 0.995:
    # for i in range(0, m):
        i = i+1
        gen_der2 = qcircuit.get_generator_der_list()
        gradient_descent(gen_der1, gen_der2, gen_step, gen_para)
        qcircuit.change_generator_para(gen_para)
        gen_der1 = gen_der2[:]
        y2.append(loss(qcircuit))
        for j in range(0, 5):
            dis_der2 = opposite(qcircuit.get_discriminator_der_list())
            gradient_descent(dis_der1, dis_der2, dis_step, dis_para)
            qcircuit.change_discriminator_para(dis_para)
            dis_der1 = dis_der2[:]
        y3.append(loss(qcircuit))
        print(fidelity(qcircuit))
        y1.append(fidelity(qcircuit))
    generate_state = qcircuit.get_gen_state()
    return generate_state
    # x = np.arange(0, i, 1)
    # plt.xlabel("iteration")
    # plt.ylabel("fidelity")
    # plt.plot(x, y1, 'd')
    # plt.savefig('./fidelity.png')
    # plt.show()


if __name__ == '__main__':
    img1 = cv2.imread('./2.jpg', 0)
    # plt.imshow(img1, cmap="gray")
    # plt.show()
    y = np.zeros((28, 28), dtype=float)
    a = {}
    ket = np.array([[0.0], [0.0], [0.0], [0.0]])
    generate_states = np.array([[0.0], [0.0], [0.0], [0.0]])
    generate_image = []
    for i in range(0, 28, 2):
        for j in range(0, 28, 2):
            print(i, j)
            light_up = img1[i][j] + 1
            light_down = img1[i+1][j] + 1
            right_up = img1[i][j+1] + 1
            right_down = img1[i+1][j+1] + 1
            sum_square = light_up**2 + light_down**2 + right_up**2 + right_down**2
            sum_square = sum_square ** 0.5
            light_up /= sum_square
            light_down /= sum_square
            right_up /= sum_square
            right_down /= sum_square
            # ket = np.array([[0.0], [0.0], [0.0], [0.0]])
            ket[0], ket[1], ket[2], ket[3] = light_up, light_down, right_up, right_down
            print(ket[0], ket[1], ket[2], ket[3])

            # 检测重复
            a1, a2, a3, a4 = light_up, light_down, right_up, right_down
            input_key = a1, a2, a3, a4

            if a.get(input_key) != None:
                output_key = a.get(input_key)
                generate_states[0][0] = output_key[0]
                generate_states[1][0] = output_key[1]
                generate_states[2][0] = output_key[2]
                generate_states[3][0] = output_key[3]
            else:
                # ket of target state
                cir = QC.QCircuitSimulator(ket, 1, 1, 2, "no_noise")
                generate_states = iteration(cir, 100)
                # print(generate_states[0][0],generate_states[0][1])
            output_key = generate_states[0][0], generate_states[1][0], generate_states[2][0], generate_states[3][0]
            a[input_key] = output_key

            y[i][j] = float(abs(generate_states[0][0]))*sum_square - 1
            # print(sum_square, generate_states[0][0], y[i][j])

            y[i+1][j] = float(abs(generate_states[1][0]))*sum_square - 1
            y[i][j+1] = float(abs(generate_states[2][0]))*sum_square - 1
            y[i+1][j+1] = float(abs(generate_states[3][0]))*sum_square - 1
            # print(y)
            if y[i][j] > 255:
                y[i][j] = 255
            if y[i+1][j] > 255:
                y[i+1][j] = 255
            if y[i][j+1] > 255:
                y[i][j+1] = 255
            if y[i+1][j+1] > 255:
                y[i+1][j+1] = 255
            output_key = y[i][j], y[i+1][j], y[i][j+1], y[i+1][j+1]
            a[input_key] = output_key

    path_name = "./experiment/figure_generate/y.npy"
    a = np.array(y)
    np.save(path_name, a)

    plt.imshow(y, cmap="gray")
    plt.savefig('./experiment/figure_generate/g2.png')  # 保存图片
    plt.show()

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






