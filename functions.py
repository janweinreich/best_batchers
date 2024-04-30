import random
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from botorch_ext import optimize_acqf_discrete_modified
import matplotlib.pyplot as plt


from datasets import init_directaryl, init_formed
from BO_utils import update_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf_discrete

random.seed(666)
torch.manual_seed(666)
np.random.seed(666)



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


def bo_inner(model, bounds_norm, q,
             X_train, y_train, X_pool, y_pool,
             yield_thr=99.0):

    sampler = SobolQMCNormalSampler(1024, seed=666)

    # Set up aqf
    qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
    X_candidate, _ = optimize_acqf_discrete(
      acq_function=qNEI,
      bounds=bounds_norm,
      q=q,
      choices=torch.tensor(X_pool),
      unique=True,
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
      #print("We found some good candidate! :)")
        pass
    else:
      #print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
      X_train = np.vstack([X_train, X_candidate])
      y_train = np.concatenate([y_train, y_candidate])
      model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

    #print(y_candidate)
    return success, n_experiments, model, X_train, y_train, X_pool, y_pool, float(max(y_candidate))

def bo_varying_q(model, qarr, X_train, X_pool, iteration):

    n_best = 100
    measure = []
    for q in qarr:
        sampler = SobolQMCNormalSampler(1024, seed=666)

        # Set up aqf
        qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)

        best_candidates, best_acq_values = optimize_acqf_discrete_modified(qNEI, q, torch.tensor(X_pool), n_best, unique=True)
        best_candidates = best_candidates.view(n_best, q, best_candidates.shape[2])
        best_acq_values = best_acq_values.view(n_best, q)

        best_acq_values_norm = best_acq_values.sum(axis=1)


        coef = compute_outlier_measure_single(best_acq_values_norm)
        measure.append( coef)
        print(q,  coef)

    plt.plot(qarr, measure, 'o-')
    plt.savefig(f'q_measure_{iteration}.png')

def compute_outlier_measure_single(dist):
    """
    Compute the outlier measure for the maximum value in a distribution.
    An outlier is considered based on large values only.

    Parameters:
    - dist (torch.Tensor): A 1D tensor representing a distribution.

    Returns:
    - float: The outlier measure for the distribution.
    """
    mean = torch.mean(dist)
    std = torch.std(dist)

    # Standardizing the distribution
    standardized_dist = (dist - mean) / std

    # Finding the maximum value's standardized score
    max_value_score = torch.max(standardized_dist)

    return max_value_score.item()


def bo_above(q, seed, max_iterations=100):

  model, X_train, y_train, X_pool, y_pool, bounds_norm = init_directaryl(seed)

  # Count experiments
  n_experiments = 0

  # Count iterations
  n_iter = 0

  for i in range(max_iterations):
    print(f'{i=} {q=} {seed=}')
    is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool, _ = bo_inner(model, bounds_norm, q, X_train, y_train, X_pool, y_pool)
    n_experiments += n_experiments_incr
    n_iter += 1
    if is_found is True:
      break

  return n_experiments, n_iter


def bo_above_flex_batch(q_arr, seed,dataset, max_iterations=100):
    if dataset == 'formed':
        model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)
    elif dataset == 'directaryl':
        model, X_train, y_train, X_pool, y_pool, bounds_norm = init_directaryl(seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # Count experiments
    n_experiments = 0
    for i in range(max_iterations):
        q = q_arr[i] if i<len(q_arr) else q_arr[-1]
        print(f'{i=} {q=} {seed=}')
        is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool, _ = bo_inner(model, bounds_norm, q, X_train, y_train, X_pool, y_pool)
        n_experiments += n_experiments_incr
        if is_found is True:
            break

    return n_experiments, i+1


def bo_above_adaptive_batch(q0, seed,dataset, max_iterations=100):
    if dataset == "formed":
        model, X_train, y_train, X_pool, y_pool, bounds_norm = init_formed(seed)
    elif dataset == "directaryl":
        model, X_train, y_train, X_pool, y_pool, bounds_norm = init_directaryl(seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model, X_train, y_train, X_pool, y_pool, bounds_norm = init_directaryl(seed)

    # Count experiments
    n_experiments = 0
    dy = None
    dy_old = 0.0
    y_best_candidate_old = 0.0

    for i in range(max_iterations):
        if i==0:
            q = q0
        else:
            if (0 < dy < dy_old) and (q > 2):
                q -= 1
            dy_old = dy
            y_best_candidate_old = y_best_candidate

        is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool, y_best_candidate = bo_inner(model, bounds_norm, q, X_train, y_train, X_pool, y_pool)
        dy = y_best_candidate - y_best_candidate_old
        print(f'{i=} {q=} {seed=} {y_best_candidate=} {dy=} {dy_old=}')
        n_experiments += n_experiments_incr
        if is_found is True:
            break

    return n_experiments, i+1






#if __name__ == '__main__':
    """
    Example how to load and fit the FORMED dataset
    
    FORMED_DATASET  = formed(new_parse=True, SMILES_MODE=True)
    X_init, y_init, X_holdout, y_holdout, smiles_init, smiles_holdout = FORMED_DATASET.get_init_holdout_data(666)
    y = y_holdout.flatten().detach().numpy()

    print("Molecule with largest gap:", smiles_holdout[np.argmax(y)], max(y))
    print("Molecule with smallest gap:", smiles_holdout[np.argmin(y)], min(y))

    model, _ = update_model(
        X_init,
        y_init,
        bounds_norm=None,
        kernel_type="Tanimoto",
        fit_y=False,
        FIT_METHOD=True,
        surrogate="GP",
    )
    y_pred = model(X_holdout).mean.detach().numpy().flatten()
    # compute the pearson correlation with scikit-learn
    from sklearn.metrics import r2_score
    r2 = r2_score(y_holdout, y_pred)
    print(f"R2: {r2}")

    # make a scatter plot
    plt.scatter(y_holdout, y_pred, alpha=0.004)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted')
    plt.savefig('true_vs_predicted_spham.png')
    plt.show()
    """
