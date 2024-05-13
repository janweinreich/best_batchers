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


max_batch_size = 8
n_seeds = 15          
max_iterations = 20


target_thr= 15.0  #99.5, Molecule with largest gap: FC(F)OC(F)C(F)(F)F 16.38479
n_best = 10000
qarr = np.arange(1, max_batch_size+1, 1)


NEW_DYNAMIC = True
NEW_static = True


if NEW_DYNAMIC:

    # qdyn = np.array([6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1])
    inter_med_alphas = []
    inter_med_n_experiments = []
    inter_med_time = []
    inter_med_best = []
    for seed in range(n_seeds):
        qdyn = 8
        print(f"seed: {seed}")
        model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)
        best_y_now = max(y_train)[0]
        n_experiments = 0
        timing = 0

        for i in range(max_iterations):

            print(f"Best value: {max(y_train)}")
            sampler = SobolQMCNormalSampler(1024, seed=666)

            # Set up aqf
            qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
            X_candidate, best_acq_values = optimize_acqf_discrete_modified(
            qNEI,
            q=qdyn,
            choices=torch.tensor(X_pool),
            n_best=n_best,
            unique=True)

            X_candidate = X_candidate.view(n_best, qdyn, X_candidate.shape[2])
            best_acq_values = best_acq_values.view(n_best, qdyn)

            best_acq_values_norm = (
                best_acq_values.mean(axis=1).mean().item()
                / best_acq_values.std(axis=1).mean().item()
            )
            print({"q": qdyn, "i": i, "best_acq_values_norm": best_acq_values_norm})

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

            n_experiments += qdyn
            timing += 1

            if success:
                break

            if max(y_train)[0] > best_y_now:
                if qdyn > 1:
                    qdyn = qdyn - 1
                best_y_now = max(y_train)[0]

        inter_med_n_experiments.append(n_experiments)
        inter_med_time.append(timing)
        inter_med_best.append(max(y_train)[0])

    dump2pkl({"exp_count": inter_med_n_experiments, "timing_count": inter_med_time, "all_y_best": inter_med_best}, "results_dynamic.pkl")


if NEW_static:
    alphas_q = []
    exp_count = []
    timing_count = []
    all_y_best = []
    for q0 in qarr:
        inter_med_alphas = []
        inter_med_n_experiments = []
        inter_med_time = []
        inter_med_best = []

        for seed in range(n_seeds):
            print(f"q0: {q0}, seed: {seed}")
            model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)

            n_experiments = 0
            timing = 0

            for i in range(max_iterations):

                print(f"Best value: {max(y_train)}")
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

                # Remove from pool
                X_pool = X_pool[indices_keep]
                y_pool = y_pool[indices_keep]

                # If we got good performance, we are done
                success = any(y_candidate > target_thr)

                # print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
                X_train  = np.vstack([X_train, X_candidate])
                y_train  = np.concatenate([y_train, y_candidate])
                model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

                inter_med_alphas.append(best_acq_values_norm)
                n_experiments += q0
                timing += 1

                if success:
                    break

            inter_med_n_experiments.append(n_experiments)
            inter_med_time.append(timing)
            inter_med_best.append(max(y_train)[0])

        alphas_q.append(inter_med_alphas)
        exp_count.append(inter_med_n_experiments)
        timing_count.append(inter_med_time)
        all_y_best.append(inter_med_best)

    dump2pkl({"alphas_q": alphas_q, "exp_count": exp_count, "timing_count": timing_count, "all_y_best": all_y_best}, "results_static.pkl")

else:

    results_dyn = loadpkl("results_dynamic.pkl")
    exp_count_dyn = np.array(results_dyn["exp_count"]).mean()
    exp_count_std_dyn = np.array(results_dyn["exp_count"]).std()
    timing_count_dyn = np.array(results_dyn["timing_count"]).mean()
    timing_count_std_dyn = np.array(results_dyn["timing_count"]).std()
    all_y_best_dyn = np.array(results_dyn["all_y_best"]).mean()
    all_y_best_std_dyn = np.array(results_dyn["all_y_best"]).std()



    #####
    results = loadpkl("results.pkl")

    alphas_q = results["alphas_q"]
    exp_count = np.array(results["exp_count"]).mean(axis=1)
    exp_count_std = np.array(results["exp_count"]).std(axis=1)
    timing_count = np.array(results["timing_count"]).mean(axis=1)
    timing_count_std = np.array(results["timing_count"]).std(axis=1)
    all_y_best = np.array(results["all_y_best"]).mean(axis=1)
    all_y_best_std = np.array(results["all_y_best"]).std(axis=1)

    fig, ax = plt.subplots()
    # plot the pareto front
    ax.plot(exp_count, timing_count, label="tradeoff", ls="--")
    pdb.set_trace()
    # for exp, time, q in zip(exp_count, timing_count, qarr):

    plt.errorbar(
        exp_count,
        timing_count,
        xerr=exp_count_std,
        yerr=timing_count_std,
        fmt="o"
    )


    plt.errorbar(exp_count_dyn, timing_count_dyn, xerr=exp_count_std_dyn, yerr=timing_count_std_dyn, fmt="o", label="dynamic")

    ax.set_xlabel("Number of experiments")
    ax.set_ylabel("Time")
    plt.legend()
    plt.savefig("pareto_front.png")

    pdb.set_trace()

    """
    alphas_q = np.load("alphas.npy")
    alphas_q = alphas_q.reshape((len(qarr), n_seeds, max_iterations))
    alphas_q = alphas_q.mean(axis=1)

    fig, ax = plt.subplots()
    # plot the average alphas over iterations
    for q in range(len(qarr)):
        ax.plot(alphas_q[q], label=f"q={qarr[q]}")
    plt.legend()
    plt.savefig("alphas_2.png")
    """
