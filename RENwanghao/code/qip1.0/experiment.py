from Quantum_Circuit import AdversarialQCircuit as QC
import numpy as np
from matplotlib import pyplot as plt
import copy
import pickle
from iRprop import iteration


Epochs = 100


def save_figure(noise_type, i, d_loss, g_loss, fidelity, m):
    path_name = "./experiment/2bit/" + str(i) + "/" + noise_type + "/"

    file_name = path_name + "d_loss.npy"
    a = np.array(d_loss)
    np.save(file_name, a)
    # a = np.load(file_name)
    # graphTable = a.tolist()
    file_name = path_name + "g_loss.npy"
    a = np.array(g_loss)
    np.save(file_name, a)
    file_name = path_name + "fidelity.npy"
    a = np.array(fidelity)
    np.save(file_name, a)

    x = np.arange(0, m, 1)
    #plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("d_loss")
    plt.plot(x, d_loss, 'd')
    file_name = path_name + "d_loss.png"
    plt.savefig(file_name)
    plt.close()
    #plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("g_loss")
    plt.plot(x, g_loss, 'd')
    file_name = path_name + "g_loss.png"
    plt.savefig(file_name)
    plt.close()
    #plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("fidelity")
    plt.plot(x, fidelity, 'd')
    file_name = path_name + "fidelity.png"
    plt.savefig(file_name)
    plt.close()


def save_cir(type, i, cir):
    file_name = "./experiment/2bit/" + str(i) + "/" + type + "/cir"
    c = pickle.dumps(cir)
    with open(file_name, "wb") as f:
        f.write(c)
    # f = open(file_name, "rb")
    # obj = pickle.load(f)
    # f.close()



for i in range(1, 11):
    cir = QC.QCircuitSimulator(1, 1, 1, 2, "Without_Noise")  # noise_name = Bit_Flip,Phase_Flip, Bit_Phase_Flip, Depolarizing_Channel
    save_cir("init", i, cir)

    c = copy.deepcopy(cir)
    fidelity, g_loss, d_loss = iteration(c, Epochs)
    save_cir("without_noise", i, c)
    save_figure("without_noise", i, d_loss, g_loss, fidelity, Epochs)

    c = copy.deepcopy(cir)
    c.noise_type = "Bit_Flip"
    fidelity, g_loss, d_loss = iteration(c, Epochs)
    save_cir("bit_flip", i, c)
    save_figure("bit_flip", i, d_loss, g_loss, fidelity, Epochs)

    c = copy.deepcopy(cir)
    c.noise_type = "Phase_Flip"
    fidelity, g_loss, d_loss = iteration(c, Epochs)
    save_cir("phase_flip", i, c)
    save_figure("phase_flip", i, d_loss, g_loss, fidelity, Epochs)

    c = copy.deepcopy(cir)
    c.noise_type = "Bit_Phase_Flip"
    fidelity, g_loss, d_loss = iteration(c, Epochs)
    save_cir("bit_phase_flip", i, c)
    save_figure("bit_phase_flip", i, d_loss, g_loss, fidelity, Epochs)
    
    c = copy.deepcopy(cir)
    c.noise_type = "Depolarizing_Channel"
    fidelity, g_loss, d_loss = iteration(c, Epochs)
    save_cir("depolarizing_channel", i, c)
    save_figure("depolarizing_channel", i, d_loss, g_loss, fidelity, Epochs)

