import random
import numpy as np
random.seed(777)
np.random.seed(777)
import torch
from botorch.exceptions import InputDataWarning
import warnings
from functions import bo_above, bo_above_flex_batch
from plots import plot_results


# To ignore a specific UserWarning about tensor construction
warnings.filterwarnings('ignore', message='.*To copy construct from a tensor.*', category=UserWarning)
# To ignore a specific InputDataWarning about input data not being standardized
warnings.filterwarnings('ignore', message='.*Input data is not standardized.*', category=UserWarning)
warnings.filterwarnings('ignore', category=InputDataWarning)

# Set device: Apple/NVIDIA/CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
dtype = torch.float


max_batch_size = 10  # 10
n_seeds = 10         # 10
max_iterations = 100  # 100


def get_baseline(max_batch_size, n_seeds, max_iterations):
    q_arr = range(2, max_batch_size+1)
    timings_all = np.zeros((len(q_arr), n_seeds, 2))
    for q in q_arr:
        timings_all[q] = [bo_above(q=q, seed=seed, max_iterations=max_iterations) for seed in range(n_seeds)]
    timings_all_mean = timings_all.mean(axis=1)
    return timings_all, timings_all_mean


if False:
    timings_all_baseline, timings_all_baseline_mean = get_baseline(max_batch_size, n_seeds, max_iterations)
    np.save("timings_all_baseline.npy", timings_all_baseline)
    print(timings_all_baseline_mean)
    #plot_results(q_arr, timings_all_baseline)





q_arr = [2, 2] # np.arange(10,1,-1)

timings_all = np.zeros((n_seeds, 2))
for seed in range(n_seeds):
    timings_all[seed] = bo_above_flex_batch(q_arr, seed=seed, max_iterations=max_iterations)
np.save("timings_all_compare.npy", timings_all)
print(timings_all)
timings_all_mean = timings_all.mean(axis=0)
timings_exps = timings_all_mean
print("2nd output", timings_exps)
