import random
import numpy as np
random.seed(777)
np.random.seed(777)
import torch
from botorch.exceptions import InputDataWarning
import warnings
from functions import init_directaryl
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
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


max_batch_size = 8   # 10
n_seeds = 20          # 10
max_iterations = 10  # 20


yield_thr= 16.0  #99.5, Molecule with largest gap: FC(F)OC(F)C(F)(F)F 16.38479
n_best = 10000
qarr = np.arange(2, max_batch_size+1, 1)

NEW = False

if NEW:
    alphas_q = []
    for q0 in qarr:
        inter_med_alphas = []
        for seed in range(n_seeds):
            print(f"q0: {q0}, seed: {seed}")
            model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)
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

                best_acq_values_norm = (
                    best_acq_values.mean(axis=1).mean().item()
                    / best_acq_values.std(axis=1).mean().item()
                )
                print({"q": q0, "i": i, "best_acq_values_norm": best_acq_values_norm})
                

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


                # print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
                X_train = np.vstack([X_train, X_candidate])
                y_train = np.concatenate([y_train, y_candidate])
                model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

                inter_med_alphas.append(best_acq_values_norm)

        alphas_q.append(inter_med_alphas)

    #pdb.set_trace()
    alphas_q = np.array(alphas_q)
    #save the average alphas
    np.save("alphas.npy", alphas_q)
    fig, ax = plt.subplots()
    # plot the average alphas over iterations
    ax.plot(alphas_q)

    plt.savefig("alphas.png")

else:
    alphas_q = np.load("alphas.npy")
    alphas_q = alphas_q.reshape((len(qarr), n_seeds, max_iterations))
    alphas_q = alphas_q.mean(axis=1)

    fig, ax = plt.subplots()
    # plot the average alphas over iterations
    for q in range(len(qarr)):
        ax.plot(alphas_q[q], label=f"q={qarr[q]}")
    plt.legend()
    plt.savefig("alphas_2.png")
    
    pdb.set_trace()