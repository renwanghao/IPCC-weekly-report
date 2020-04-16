import numpy as np
from QuantumCircuits.quantumcircuits import random_ListTheta, Vtheta, Vtheta2
from QuantumCircuits.model import Generator, Discriminator, get_zero_state, get_real_state, compute_fidelity, compute_cost
import matplotlib.pyplot as plt

# 参数
size = 2
system_size1 = 2
epochs = 1000

#  画图


def show_figure(fidelities, losses):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Fidelity between real and fake states')
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Loss')
    plt.show()
# def construct_gen(size,lst):
#     '''
#         the function to construct quantum circuit of generator
#     :param qc:
#     :param size:
#     :return:
#     '''
#     '''
#         功能构成一个量子线路，首先需要一个量子线路类的 实例qc，和size
#     '''
#     theta_list = lst
#     matrix_rep, theta_list = Vtheta(theta_list)
#
#     return matrix_rep, theta_list
#
#
# def construct_dis(size,lst):
#
#     phi_num = size * 15
#     phi_list = lst
#     matrix_rep, phi_list = Vtheta2(phi_list)
#
#     return matrix_rep,phi_list


# a, b = construct_qcircuit(size)


def main():

    # 画图用
    ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
    ay = []  # 定义一个 y 轴的空列表用来接收动态的数据

    # define generator
    gen_theta = random_ListTheta(15)  # 生成theta的随机参数list
    gen = Generator(size, gen_theta)  # define gen，输入size和参数
    mat_gen, gen_theta = gen.qc.Vtheta1()  # 得到生成器的矩阵表示和参数
    # print(mat_gen, gen_theta) # mat_gen is generator's matrix
    # representation. gen_theta is generator's param
    # p = np.asmatrix(mat_gen).getH()
    # # t = p.
    # print(p)
    # define discriminator
    dis_phi = random_ListTheta(30)
    dis = Discriminator(size, dis_phi)
    mat_dis, dis_phi = dis.qc.Vtheta2()       # 定义判别器的线路，定义好了线路之后会返回线路的矩阵表示和参数list
    # print(mat_dis, dis_phi)

    # fidelity
    zero_state = get_zero_state(size)
    real_state = get_real_state(size)
    f = compute_fidelity(mat_gen, zero_state, real_state)
    print(f)
    fidelities = []
    losses = []

    while (f < 0.99):

        for iter in range(epochs):
            print("Epoch {}".format(iter + 1))
            # update generator's param ,theta
            gen_theta_new = gen.update_theta(
                mat_gen,
                gen_theta,
                mat_dis,
                dis_phi,
                real_state)  # dis输入的是矩阵，real_state是 size方 *1 的向量
            gen = Generator(size, gen_theta_new)
            mat_gen_new, gen_theta_new = gen.qc.Vtheta1()
            print(
                "Loss after generator step: {}".format(
                    compute_cost(
                        mat_gen_new,
                        gen_theta_new,
                        mat_dis,
                        dis_phi,
                        real_state)))

            # update discriminator's param ,phi
            dis_phi_new = dis.update_phi(
                mat_gen, gen_theta, mat_dis, dis_phi, real_state)
            dis = Discriminator(size, dis_phi_new)
            mat_dis_new, dis_phi_new = dis.qc.Vtheta2()

            print(
                "Loss after discriminator step: {}".format(
                    compute_cost(
                        mat_gen_new,
                        gen_theta_new,
                        mat_dis_new,
                        dis_phi_new,
                        real_state)))

            cost = compute_cost(
                mat_gen_new,
                gen_theta_new,
                mat_dis_new,
                dis_phi_new,
                real_state)

            fidelity = compute_fidelity(mat_gen_new, zero_state, real_state)
            losses.append(cost)
            fidelities.append(fidelity)

            print(
                "Fidelity between real and fake state: {}".format(fidelity))
            print("==================================================")
            gen_theta = gen_theta_new
            dis_phi = dis_phi_new

            # 实时画图
            # ax.append(iter)  # 添加 迭代次数 i 到 x 轴的数据中
            # ay.append(fidelity)  # 添加 i 的平方到 y 轴的数据中
            # plt.clf()  # 清除之前画的图
            # plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
            # plt.pause(0.1)  # 暂停一秒
            # plt.ioff()  # 关闭画图的窗口

        show_figure(fidelities, losses)
    print('end')


if __name__ == '__main__':

    main()
