import numpy as np
from matplotlib import pyplot as plt

def compute_cost(rt, ot, n_exp, n_iter):
    total_cost = 0.0
    total_cost += n_iter  # Baseline experiment cost (per iteration)
    total_cost += rt * n_iter # Retraining cost (per iteration)
    total_cost += (n_exp - n_iter) * ot # Sum of the overheads
    return total_cost


def matrix_plot(timings_exps, rt_arr, ot_arr):
    x, y = np.meshgrid(rt_arr, ot_arr)
    tt_arr = compute_cost(x, y, n_exp, n_iter)
    plt.xlabel("Retraining cost %")
    plt.ylabel("Overhead cost %")
    plt.pcolormesh(rt_arr, ot_arr, tt_arr)
    plt.title('Total time as a function of %overhead and %training')
    plt.colorbar()
    plt.show()

def plot_results(q_arr, timings_exps):

    rt_arr = np.linspace(0.1,1.0,5) # time of retraining as % of experiment baseline time
    ot_arr = np.linspace(0.1,1.0,5) # overhead time per experiment as % of experiment baseline time

    for n_exp, n_iter in timings_exps :
        x, y = np.meshgrid(rt_arr, ot_arr)
        tt_arr = compute_cost(x, y, n_exp, n_iter)
        plt.xlabel("Retraining cost %")
        plt.ylabel("Overhead cost %")
        plt.pcolormesh(rt_arr, ot_arr, tt_arr)
        plt.title('Total time as a function of %overhead and %training')
        plt.colorbar()
        plt.show()

    z = np.zeros((len(q_arr),len(rt_arr)*len(ot_arr)))
    x, y = np.meshgrid(rt_arr, ot_arr)
    for i, (n_exp, n_iter) in enumerate(timings_exps) :
        p = q_arr
        z[i,:] = compute_cost(x,y,n_exp,n_iter).flatten()
        plt.plot(p,z)
        plt.title('Total time as a function of q')
        plt.legend()
        plt.show()
