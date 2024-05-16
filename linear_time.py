import random
import numpy as np
random.seed(777)
np.random.seed(777)
import torch
from botorch.exceptions import InputDataWarning
import warnings
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler

from functions import *
import pdb
import pickle
import bz2
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


compress_fileopener = {True: bz2.BZ2File, False: open}
pkl_compress_ending = {True: ".pkl.bz2", False: ".pkl"}


def dump2pkl(obj, filename: str, compress: bool = False):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    compress : whether bz2 library is used for compressing the file.
    """
    output_file = compress_fileopener[compress](filename, "wb")
    pickle.dump(obj, output_file)
    output_file.close()


def loadpkl(filename: str, compress: bool = False):
    """
    Load an object from a pickle file.
    filename : name of the imported file
    compress : whether bz2 compression was used in creating the loaded file.
    """
    input_file = compress_fileopener[compress](filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def linear_time(q):
    """
    for this to make sense, use a < 1 
    """
    a = 0.1

    return 1+a*(q-1)


n_seeds = 15          
max_iterations = 20


target_thr= 15.0  #99.5, Molecule with largest gap: FC(F)OC(F)C(F)(F)F 16.38479
n_best = 10000


qmax = 10
max_time = linear_time(qmax)

for seed in [1337]:

    print(f"seed: {seed}")
    model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)
    best_y_now = max(y_train)[0]
    n_experiments = 0
    timing = 0

    for i in range(max_iterations):

        print(f"Best value: {max(y_train)}")

        # Magic happens here
        acq_values_q = []
        for q in range(1, qmax+1):

            time_spent = linear_time(q)

            sampler = SobolQMCNormalSampler(1024, seed=666)

            # Set up aqf
            qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
            X_candidate, best_acq_values = optimize_acqf_discrete_modified(
            qNEI,
            q=q,
            choices=torch.tensor(X_pool),
            n_best=n_best,
            unique=True)

            X_candidate = X_candidate.view(n_best, q, X_candidate.shape[2])
            best_acq_values = best_acq_values.view(n_best, q)

            avg_acq_val_for_best = np.mean(np.array(best_acq_values[0]))
            max_acq_val_for_best = np.max(np.array(best_acq_values[0]))

            scaling_factor = max_acq_val_for_best / max_time
            # adjust for time
            avg_acq_val_for_best = avg_acq_val_for_best - time_spent * scaling_factor
            acq_values_q.append(avg_acq_val_for_best)

        min_q = np.argmax(acq_values_q) + 1

        # Magic ends here
        pdb.set_trace()
        exit()
        sorted_indices = np.argsort(row_sums_2)[::-1]

        # See how they actually look
        X_candidate = np.array(X_candidate[0])
        indices = find_indices(X_pool, X_candidate)
        indices_keep = np.setdiff1d(np.arange(X_pool.shape[0]), indices)
        y_candidate = y_pool[indices]

        # Remove from pool
        X_pool = X_pool[indices_keep]
        y_pool = y_pool[indices_keep]

        # If we got good performance, we are done
        success = any(y_candidate > target_thr)

        # print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
        X_train  = np.vstack([X_train, X_candidate])
        y_train  = np.concatenate([y_train, y_candidate])
        model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

        if success:
            break

# dump2pkl({"exp_count": inter_med_n_experiments, "timing_count": inter_med_time, "all_y_best": inter_med_best}, "results_dynamic.pkl")
