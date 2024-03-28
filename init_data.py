import torch
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import copy as cp
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler

from BO_utils import update_model


random.seed(777)
np.random.seed(777)

def inchi_to_smiles(inchi_list):
    """
    Convert a list of InChI strings to a list of canonical SMILES strings.

    Args:
    inchi_list (list): A list of InChI strings.

    Returns:
    list: A list of canonical SMILES strings.
    """
    smiles_list = []
    for inchi in inchi_list:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)  # Append None for invalid InChI strings
    return smiles_list


class FingerprintGenerator:
    def __init__(self, nBits=512, radius=2):
        self.nBits = nBits
        self.radius = radius

    def featurize(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                fingerprints.append(np.ones(self.nBits))
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.nBits
                    )
                    fp_array = np.array(
                        list(fp.ToBitString()), dtype=int
                    )  # Convert to NumPy array
                    fingerprints.append(fp_array)
                else:
                    print(f"Could not generate a molecule from SMILES: {smiles}")
                    fingerprints.append(np.array([None]))

        return np.array(fingerprints)



def convert2pytorch(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


def check_entries(array_of_arrays):
    """
    Check if the entries of the arrays are between 0 and 1.
    Needed for for the datasets.py script.
    """

    for array in array_of_arrays:
        for item in array:
            if item < 0 or item > 1:
                return False
    return True


class directaryl:
    def __init__(self):
        # direct arylation reaction
        self.ECFP_size = 512
        self.radius = 2
        self.ftzr = FingerprintGenerator(nBits=self.ECFP_size, radius=self.radius)
        dataset_url = "https://raw.githubusercontent.com/doyle-lab-ucla/edboplus/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full.csv"
        self.data = pd.read_csv(dataset_url)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        # create a copy of the data
        data_copy = self.data.copy()
        # remove the Yield column from the copy
        data_copy.drop("Yield", axis=1, inplace=True)
        # check for duplicates
        duplicates = data_copy.duplicated().any()
        if duplicates:
            print("There are duplicates in the dataset.")
            exit()

        self.data["Base_SMILES"] = inchi_to_smiles(self.data["Base_inchi"].values)
        self.data["Ligand_SMILES"] = inchi_to_smiles(self.data["Ligand_inchi"].values)
        self.data["Solvent_SMILES"] = inchi_to_smiles(self.data["Solvent_inchi"].values)
        col_0_base = self.ftzr.featurize(self.data["Base_SMILES"])
        col_1_ligand = self.ftzr.featurize(self.data["Ligand_SMILES"])
        col_2_solvent = self.ftzr.featurize(self.data["Solvent_SMILES"])
        col_3_concentration = self.data["Concentration"].to_numpy().reshape(-1, 1)
        col_4_temperature = self.data["Temp_C"].to_numpy().reshape(-1, 1)
        self.X = np.concatenate(
            [
                col_0_base,
                col_1_ligand,
                col_2_solvent,
                col_3_concentration,
                col_4_temperature,
            ],
            axis=1,
        )
        self.experiments = np.concatenate(
            [
                self.data["Base_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Ligand_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Solvent_SMILES"].to_numpy().reshape(-1, 1),
                self.data["Concentration"].to_numpy().reshape(-1, 1),
                self.data["Temp_C"].to_numpy().reshape(-1, 1),
                self.data["Yield"].to_numpy().reshape(-1, 1),
            ],
            axis=1,
        )

        self.y = self.data["Yield"].to_numpy()
        self.all_ligands = self.data["Ligand_SMILES"].to_numpy()
        self.all_bases = self.data["Base_SMILES"].to_numpy()
        self.all_solvents = self.data["Solvent_SMILES"].to_numpy()
        unique_bases = np.unique(self.data["Base_SMILES"])
        unique_ligands = np.unique(self.data["Ligand_SMILES"])
        unique_solvents = np.unique(self.data["Solvent_SMILES"])
        unique_concentrations = np.unique(self.data["Concentration"])
        unique_temperatures = np.unique(self.data["Temp_C"])

        max_yield_per_ligand = np.array(
            [
                max(self.data[self.data["Ligand_SMILES"] == unique_ligand]["Yield"])
                for unique_ligand in unique_ligands
            ]
        )

        self.worst_ligand = unique_ligands[np.argmin(max_yield_per_ligand)]
        self.best_ligand = unique_ligands[np.argmax(max_yield_per_ligand)]

        self.where_worst_ligand = np.array(
            self.data.index[self.data["Ligand_SMILES"] == self.worst_ligand].tolist()
        )

        self.feauture_labels = {
            "names": {
                "bases": unique_bases,
                "ligands": unique_ligands,
                "solvents": unique_solvents,
                "concentrations": unique_concentrations,
                "temperatures": unique_temperatures,
            },
            "ordered_smiles": {
                "bases": self.data["Base_SMILES"],
                "ligands": self.data["Ligand_SMILES"],
                "solvents": self.data["Solvent_SMILES"],
                "concentrations": self.data["Concentration"],
                "temperatures": self.data["Temp_C"],
            },
        }


class Evaluation_data:
    def __init__(
        self
    ):
        self.get_raw_dataset()

        rep_size = self.X.shape[1]
        self.bounds_norm = torch.tensor([[0] * rep_size, [1] * rep_size])
        self.bounds_norm = self.bounds_norm.to(dtype=torch.float32)

        if not check_entries(self.X):
            print("###############################################")
            print(
                "Entries of X are not between 0 and 1. Adding MinMaxScaler to the pipeline."
            )
            print("###############################################")

            self.scaler_X = MinMaxScaler()
            self.X = self.scaler_X.fit_transform(self.X)

    def get_raw_dataset(self):
        # https://github.com/doyle-lab-ucla/edboplus/blob/main/examples/publication/BMS_yield_cost/data/PCI_PMI_cost_full_update.csv

        BMS = directaryl()
        self.data = BMS.data
        self.experiments = BMS.experiments
        self.X, self.y = BMS.X, BMS.y

        self.all_ligands = BMS.all_ligands
        self.all_bases = BMS.all_bases
        self.all_solvents = BMS.all_solvents

        self.best_ligand = BMS.best_ligand
        self.worst_ligand = BMS.worst_ligand
        self.where_worst_ligand = BMS.where_worst_ligand
        self.feauture_labels = BMS.feauture_labels


    def get_init_holdout_data(self, SEED):
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Reduce the number of initial experiments to 48
        indices_init = np.random.choice(self.where_worst_ligand[:200], size=48, replace=False)
        exp_init = self.experiments[indices_init]
        indices_holdout = np.setdiff1d(np.arange(len(self.y)), indices_init)

        np.random.shuffle(indices_init)
        np.random.shuffle(indices_holdout)

        X_init, y_init = self.X[indices_init], self.y[indices_init]
        X_holdout, y_holdout = self.X[indices_holdout], self.y[indices_holdout]
        exp_holdout = self.experiments[indices_holdout]

        LIGANDS_INIT = self.all_ligands[indices_init]
        LIGANDS_HOLDOUT = self.all_ligands[indices_holdout]

        X_init, y_init = convert2pytorch(X_init, y_init)
        X_holdout, y_holdout = convert2pytorch(X_holdout, y_holdout)

        return (
            X_init,
            y_init,
            X_holdout,
            y_holdout,
            LIGANDS_INIT,
            LIGANDS_HOLDOUT,
            exp_init,
            exp_holdout,
        )

# Functions from Notebook

def find_indices(X_candidate_BO, candidates):
    """
    Identifies and returns the indices of specific candidates within a larger dataset.
    This function is particularly useful when the order of candidates returned by an
    acquisition function differs from the original dataset order.

    Args:
        X_candidate_BO (numpy.ndarray): The complete dataset or holdout set,
            typically consisting of feature vectors.
        candidates (numpy.ndarray): A subset of the dataset (e.g., a batch of
            molecules) selected by the acquisition function.

    Returns:
        numpy.ndarray: An array of indices corresponding to the positions of
            each candidate in the original dataset 'X_candidate_BO'.
    """

    indices = []
    for candidate in candidates:
        indices.append(np.argwhere((X_candidate_BO == candidate).all(1)).flatten()[0])
    indices = np.array(indices)
    return indices

# The main BO loop for fixed q and helper functions
def bo_inner(model, sampler, bounds_norm, q, X_train, y_train, X_pool, y_pool, yield_thr=99.0):
    """
    The inner loop of the BO algorithm. This function selects the next batch of
    candidates based on the qNEI acquisition function (qNoisyExpectedImprovement) and updates the model
    Args:
        model: model object
        sampler: SobolQMCNormalSampler
        bounds_norm: bounds for the ac
        q: batch size
        X_train: training data
        y_train: training labels
        X_pool: pool data
        y_pool: pool labels
        yield_thr: 99 (default)  # seems high, since the error of measurement will be much higher than 1 %

    Returns:

    """
    # Set up aqf
    qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
    X_candidate, _ = optimize_acqf_discrete(
        acq_function=qNEI,
        bounds=bounds_norm,
        q=q,
        choices=torch.tensor(X_pool),
        unique=True,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        sequential=False,
    )

    # See how they actually look
    X_candidate = np.array(X_candidate)
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
        print("We found some good candidate! :)")
    else:
        print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
        X_train = np.vstack([X_train, X_candidate])
        y_train = np.concatenate([y_train, y_candidate])
        model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

    print(y_candidate)
    return success, n_experiments, model, X_train, y_train, X_pool, y_pool


def init_stuff(seed=777):
    """
    Initialisation of dataset, model and splits.

    Args:
        seed: random seed (default 777)

    Returns:
        model: model object
        X_train: training data, data which is already used by the model
        y_train: training labels
        X_pool: pool data, data which could be sampled in the future
        y_pool: pool labels

    """
    # Initialize data from dataset
    DATASET = Evaluation_data()
    bounds_norm = DATASET.bounds_norm

    # Our common starting point for data
    # X_init contains 144 datapoints as tensor, feature length 1538
    # y_init contains 144 labels as tensor
    # Pool contains datapoints which can be requested / sampled by accquisition function and sampler
    # X_pool_fixed contains 1584 datapoints as tensor, feature length 1538
    # y_pool_fixed contains 1584 labels as tensor
    (
        X_init,
        y_init,
        X_pool_fixed,
        y_pool_fixed,
        _,
        _,
        _,
        _,
    ) = DATASET.get_init_holdout_data(seed)


    # Construct initial shitty model
    model, _ = update_model(
        X_init, y_init, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True
    )

    # Copy things to avoid problems later
    X_train = np.copy(X_init)
    y_train = np.copy(y_init)
    X_pool = np.copy(X_pool_fixed)
    y_pool = np.copy(y_pool_fixed)

    return model, X_train, y_train, X_pool, y_pool

def bo_above(q, seed, max_iterations=100):
    """
    Runs the BO loop with a fixed batch size q.
    Number of experiment is counter for the amount of experiments conducted.
    ? Number of iterations ?
    q: Batch size
    seed: random seed
    max_iterations:

    Returns: number of experiments, number of iterations

    """
    model, X_train, y_train, X_pool, y_pool = init_stuff(seed)

    # Count total experiments needed
    n_experiments = 0

    # Count iterations of the BO cycle
    # Exapmle: With a batch size of 10 and n_iter of 5 we would have 50 additional experiments
    n_iter = 0
    # Is not equal to max_iterations if we find a good candidate before

    for i in range(max_iterations):
        is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool = bo_inner(model, sampler, bounds_norm, q,
                                                                                         X_train, y_train, X_pool,
                                                                                         y_pool)
        n_experiments += n_experiments_incr
        n_iter += 1
        if is_found is True:
            break

    return n_experiments, n_iter


DATASET = Evaluation_data()
bounds_norm = DATASET.bounds_norm

(
    X_init,
    y_init,
    X_pool_fixed,
    y_pool_fixed,
    LIGANDS_INIT,
    LIGANDS_HOLDOUT,
    exp_init,
    exp_holdout,
) = DATASET.get_init_holdout_data(777)

print("Current state")



# q_arr = range(2, max_batch_size+1)
#
# timings_all = np.zeros((n_seeds, len(q_arr), 2))
# for seed in range(n_seeds):
#   timings_all[seed] = [bo_above(q=q, seed=seed, max_iterations=max_iterations) for q in q_arr]
#
# timings_all_mean = timings_all.mean(axis=0)
# timings_exps = timings_all_mean
#
#
# rt_arr = np.linspace(0.1,1.0,5) # time of retraining as % of experiment baseline time
# ot_arr = np.linspace(0.1,1.0,5) # overhead time per experiment as % of experiment baseline time

def compute_cost(rt, ot, n_exp, n_iter):
  total_cost = 0.0
  total_cost += n_iter  # Baseline experiment cost (per iteration)
  total_cost += rt * n_iter # Retraining cost (per iteration)
  total_cost += (n_exp - n_iter) * ot # Sum of the overheads
  return total_cost

# for n_exp, n_iter in timings_exps :
#   x, y = np.meshgrid(rt_arr, ot_arr)
#   tt_arr = compute_cost(x, y, n_exp, n_iter)
#   plt.xlabel("Retraining cost %")
#   plt.ylabel("Overhead cost %")
#   plt.pcolormesh(rt_arr, ot_arr, tt_arr)
#   plt.title('Total time as a function of %overhead and %training')
#   plt.colorbar()
#   plt.show()
#
#   z = np.zeros((len(q_arr), 25))
#   x, y = np.meshgrid(rt_arr, ot_arr)
#   for i, (n_exp, n_iter) in enumerate(timings_exps):
#       p = q_arr
#       z[i, :] = compute_cost(x, y, n_exp, n_iter).flatten()
#   plt.plot(p, z)
#   plt.title('Total time as a function of q')
#   plt.legend()
#   plt.show()



# Code from Notebook

NUM_RESTARTS = 20
RAW_SAMPLES = 512

# Selection of next datapoints based on aquisition function
sampler = SobolQMCNormalSampler(1024)

# Returns number of iterations and number of experiments
#bo_above(q=3, seed=666, max_iterations=5)

# Get baseline results with fixed q
max_batch_size = 10  # 10
n_seeds = 10         # 10
max_iterations = 100  # 100

def q_exp_decay(mean_value_acq_function,max_batch_size, min_batch_size=3):
    q_arr = np.exp(-mean_value_acq_function)*max_batch_size
    if int(q_arr) < min_batch_size:
        q_arr = min_batch_size
    return int(q_arr)


# BO loop but with q depending on iteration number
def bo_above_flex_batch(q_arr, seed, max_iterations=100):
    model, X_train, y_train, X_pool, y_pool = init_stuff(seed)

    # Count experiments
    n_experiments = 0

    for i in range(max_iterations):

        #best_observed_yield = max(y_train)[0]
        #q = int(99.9 - best_observed_yield)/max_batch_size
        #q = q_arr[i] if i < len(q_arr) else q_arr[-1]
        if i == 0:
            q = 10
        else:
            q = q_exp_decay(aqf_values.mean(), max_batch_size)
        is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool, aqf_values = bo_inner(model, sampler, bounds_norm, q,
                                                                                         X_train, y_train, X_pool,
                                                                                         y_pool)
        n_experiments += n_experiments_incr
        if is_found is True:
            print(f"Found in iteration {i} with {n_experiments} experimnets! :)")
            break

    return n_experiments, i + 1

# Try different ways to change q

# q_arr = np.arange(10,1,-2)
# q_arr = np.arange(7,1,-1)
q_arr = np.arange(5,1,-1)

n_seeds = 10         # 10
max_iterations = 100  # 100

timings_all = np.zeros((n_seeds, 2))

for seed in range(n_seeds):
  timings_all[seed] = bo_above_flex_batch(q_arr, seed=seed, max_iterations=max_iterations)
  df = pd.DataFrame(timings_all, columns=['timing', 'iterations'])
  print(df)
  df.to_csv('exp_decay_5torest.csv', index=None)

print(timings_all)
print('ENDE')
