import random
import numpy as np
random.seed(777)
np.random.seed(777)
import torch
from botorch.exceptions import InputDataWarning
import warnings
from functions import bo_varying_q
from plots import plot_results
from functions import init_stuff
import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler

from functions import *

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


max_batch_size = 10   # 10
n_seeds = 10          # 10
max_iterations = 40  # 100


yield_thr=99.5

qarr = [2] #np.arange(2, 15, 1)

n_best = 100

average_alphas = []
for q0 in qarr:
    model, X_train, y_train, X_pool, y_pool, bounds_norm = init_stuff(777)
    for i in range(max_iterations):

        sampler = SobolQMCNormalSampler(1024, seed=666)

        # Set up aqf
        qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
        X_candidate, best_acq_values = optimize_acqf_discrete_modified(
        qNEI,
        q=q0,
        choices=torch.tensor(X_pool),
        n_best=n_best,
        unique=True)

        X_candidate = X_candidate.view(n_best, q0, X_candidate.shape[2])
        best_acq_values = best_acq_values.view(n_best, q0)
        #pdb.set_trace()
        plt.hist(best_acq_values.mean(axis=1).numpy())
        plt.savefig(f"hist_{i}.png")
        plt.close()

        best_acq_values_norm = (
            best_acq_values.mean(axis=1).mean().item()
            / best_acq_values.std(axis=1).mean().item()
        )
        print({"q": q0, "i": i, "best_acq_values_norm": best_acq_values_norm})
        average_alphas.append(best_acq_values_norm)

        # See how they actually look
        X_candidate = np.array(X_candidate[0])
        indices = find_indices(X_pool, X_candidate)
        indices_keep = np.setdiff1d(np.arange(X_pool.shape[0]), indices)
        y_candidate = y_pool[indices]

        # We also count the number of experiments conducted
        n_experiments = y_candidate.shape[0]

        # Remove from pool
        X_pool = X_pool[indices_keep]
        y_pool = y_pool[indices_keep]

        # If we got good performance, we are done
        success = any(y_candidate > yield_thr)

        if success:
            # print("We found some good candidate! :)")
            pass
        else:
            # print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
            X_train = np.vstack([X_train, X_candidate])
            y_train = np.concatenate([y_train, y_candidate])
            model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)


average_alphas = np.array(average_alphas)
fig, ax = plt.subplots()
# plot the average alphas over iterations
ax.plot(average_alphas)

plt.savefig("alphas.png")
