import numpy as np
import torch
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import copy as cp
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.means import ConstantMean
from botorch.models import SingleTaskGP
from gpytorch.kernels import ScaleKernel
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

from botorch.exceptions import InputDataWarning
import warnings
import random
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


def batch_tanimoto_sim(
    x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Tanimoto similarity between two batched tensors, across last 2 dimensions.
    eps argument ensures numerical stability if all zero tensors are added. Tanimoto similarity is proportional to:

    (<x, y>) / (||x||^2 + ||y||^2 - <x, y>)

    where x and y may be bit or count vectors or in set notation:

    |A \cap B | / |A| + |B| - |A \cap B |

    Args:
        x1: `[b x n x d]` Tensor where b is the batch dimension
        x2: `[b x m x d]` Tensor
        eps: Float for numerical stability. Default value is 1e-6
    Returns:
        Tensor denoting the Tanimoto similarity.
    #from here https://github.com/leojklarner/gauche/blob/main/gauche/kernels/fingerprint_kernels/tanimoto_kernel.py
    """

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")

    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)

    tan_similarity = (dot_prod + eps) / (
        eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )

    return tan_similarity.clamp_min_(
        0
    )  # zero out negative values for numerical stability


class TanimotoKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Tanimoto kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

    \begin{equation*}
     k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) = \frac{\langle\mathbf{x},
     \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert^2 + \left\lVert\mathbf{x'}\right\rVert^2 -
     \langle\mathbf{x}, \mathbf{x'}\rangle}
    \end{equation*}

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        **params,
    ):
        r"""This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return batch_tanimoto_sim(x1, x2)


def update_model(
    X,
    y,
    bounds_norm,
    kernel_type="Tanimoto",
    fit_y=True,
    FIT_METHOD=True,
    surrogate="GP",
):
    """
    Update and return a Gaussian Process (GP) model with new training data.
    This function configures and optimizes the GP model based on the provided parameters.

    Args:
        X (numpy.ndarray): The training data, typically feature vectors.
        y (numpy.ndarray): The corresponding labels or values for the training data.
        bounds_norm (numpy.ndarray): Normalization bounds for the training data.
        kernel_type (str, optional): Type of kernel to be used in the GP model. Default is "Tanimoto".
        fit_y (bool, optional): Flag to indicate if the output values (y) should be fitted. Default is True.
        FIT_METHOD (bool, optional): Flag to indicate the fitting method to be used. Default is True.
        surrogate (str, optional): Type of surrogate model to be used. Default is "GP".

    Returns:
        model (botorch.models.gpytorch.GP): The updated GP model, fitted with the provided training data.
        scaler_y (TensorStandardScaler): The scaler used for the labels, which can be applied for future data normalization.

    Notes:
        The function initializes a GP model with specified kernel and fitting methods, then fits the model to the provided data.
        The 'bounds_norm' parameter is used for normalizing the training data within the GP model.
        The 'fit_y' and 'FIT_METHOD' parameters control the fitting behavior of the model.
    """

    GP_class = Surrogate_Model(
        kernel_type=kernel_type,
        bounds_norm=bounds_norm,
        fit_y=fit_y,
        FIT_METHOD=FIT_METHOD,
        surrogate=surrogate,
    )
    model = GP_class.fit(X, y)

    return model, GP_class.scaler_y


class TensorStandardScaler:
    """
    StandardScaler for tensors that standardizes features by removing the mean
    and scaling to unit variance, as defined in BoTorch.

    Attributes:
        dim (int): The dimension over which to compute the mean and standard deviation.
        epsilon (float): A small constant to avoid division by zero in case of a zero standard deviation.
        mean (Tensor, optional): The mean value computed in the `fit` method. None until `fit` is called.
        std (Tensor, optional): The standard deviation computed in the `fit` method. None until `fit` is called.

    Args:
        dim (int): The dimension over which to standardize the data. Default is -2.
        epsilon (float): A small constant to avoid division by zero. Default is 1e-9.
    """

    def __init__(self, dim: int = -2, epsilon: float = 1e-9):
        self.dim = dim
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, Y):
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        self.mean = Y.mean(dim=self.dim, keepdim=True)
        self.std = Y.std(dim=self.dim, keepdim=True)
        self.std = self.std.where(
            self.std >= self.epsilon, torch.full_like(self.std, 1.0)
        )

    def transform(self, Y):
        if self.mean is None or self.std is None:
            raise ValueError(
                "Mean and standard deviation not initialized, run `fit` method first."
            )
        original_type = None
        if isinstance(Y, np.ndarray):
            original_type = np.ndarray
            Y = torch.from_numpy(Y).float()
        Y_transformed = (Y - self.mean) / self.std
        if original_type is np.ndarray:
            return Y_transformed.numpy()
        else:
            return Y_transformed

    def fit_transform(self, Y):
        self.fit(Y)
        return self.transform(Y)

    def inverse_transform(self, Y):
        if self.mean is None or self.std is None:
            raise ValueError(
                "Mean and standard deviation not initialized, run `fit` method first."
            )
        original_type = None
        if isinstance(Y, np.ndarray):
            original_type = np.ndarray
            Y = torch.from_numpy(Y).float()
        Y_inv_transformed = (Y * self.std) + self.mean
        if original_type is np.ndarray:
            return Y_inv_transformed.numpy()
        else:
            return Y_inv_transformed


class Surrogate_Model:

    def __init__(
        self,
        kernel_type="Tanimoto",
        bounds_norm=None,
        fit_y=True,
        FIT_METHOD=True,
        surrogate="GP",
    ):
        self.kernel_type = kernel_type
        self.bounds_norm = bounds_norm
        self.fit_y = fit_y
        self.surrogate = surrogate
        self.FIT_METHOD = FIT_METHOD
        self.scaler_y = TensorStandardScaler()

    def fit(self, X_train, y_train):
        if type(X_train) == np.ndarray:
            X_train = torch.tensor(X_train, dtype=torch.float32)

        if self.fit_y:
            y_train = self.scaler_y.fit_transform(y_train)
        else:
            y_train = y_train

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)

        """
        Use BoTorch fit method
        to fit the hyperparameters of the GP and the model weights
        """

        self.kernel_type == "Tanimoto"
        kernel = TanimotoKernel()

        class InternalGP(SingleTaskGP):
            def __init__(self, train_X, train_Y, kernel):
                super().__init__(train_X, train_Y)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(kernel)

        self.gp = InternalGP(self.X_train_tensor, self.y_train_tensor, kernel)

        self.gp.likelihood.noise_constraint = gpytorch.constraints.GreaterThan(
                1e-3
            )

        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.mll.to(self.X_train_tensor)

        fit_gpytorch_model(self.mll, max_retries=50000)


        self.gp.eval()
        self.mll.eval()

        return self.gp

# %%
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
            #print("###############################################")
            #print(
            #    "Entries of X are not between 0 and 1. Adding MinMaxScaler to the pipeline."
            #)
            #print("###############################################")

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

# %% [markdown]
# ![numel.jpeg](data:image/jpeg;base64,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)

# %%
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler

NUM_RESTARTS = 20
RAW_SAMPLES = 512
sampler = SobolQMCNormalSampler(1024)


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


# %% [markdown]
# The main BO loop for fixed `q` and helper functions
# 

# %%

def bo_inner(model, sampler, bounds_norm, q, 
             X_train, y_train, X_pool, y_pool, 
             yield_thr=99.0):    
            
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
      #print("We found some good candidate! :)")
        pass
    else:
      #print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
      X_train = np.vstack([X_train, X_candidate])
      y_train = np.concatenate([y_train, y_candidate])
      model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

    #print(y_candidate)
    return success, n_experiments, model, X_train, y_train, X_pool, y_pool

 
def init_stuff(seed):
  # Initialize data from dataset
  DATASET = Evaluation_data()
  bounds_norm = DATASET.bounds_norm

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

  model, X_train, y_train, X_pool, y_pool = init_stuff(seed)

  # Count experiments
  n_experiments = 0

  # Count iterations
  n_iter = 0

  for i in range(max_iterations):
    is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool = bo_inner(model, sampler, bounds_norm, q, X_train, y_train, X_pool, y_pool)
    n_experiments += n_experiments_incr
    n_iter += 1
    if is_found is True:
      break

  return n_experiments, n_iter


# The old BO loop to check we don't screw up (and "tests")
# 

# %%
def bo_above_old(q, seed, max_iterations=100):

  # Initialize data from dataset
  DATASET = Evaluation_data()
  bounds_norm = DATASET.bounds_norm

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

  # Start a timer
  fixed_time = 1
  overhead_factor = 0.5
  overhead_time = overhead_factor * fixed_time
  training_time = 2
  total_time = 0

  # Count experiments
  n_experiments = 0

  # Count iterations
  n_iter = 0

  for _ in range(max_iterations):

    # Add fixed time
    total_time += fixed_time

    # Select batch size, for now dummy passed
    # Here is where the fun begins! Including the calls below

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

    # See how selected experiments look at the moment
    y_pred = model.posterior(X_candidate).mean.detach().flatten().numpy()
    y_std = np.sqrt(model.posterior(X_candidate).variance.detach().flatten().numpy())

    # See how they actually look
    X_candidate = np.array(X_candidate)
    indices = find_indices(X_pool, X_candidate)
    indices_keep = np.setdiff1d(np.arange(X_pool.shape[0]), indices)
    y_candidate = y_pool[indices]

    # We have sampled the candidates, so we pay the overhead time n-1 times
    total_time += overhead_time*(y_candidate.shape[0] -1)

    # We also count the number of experiments conducted
    n_experiments += y_candidate.shape[0]
    n_iter += 1

    # Remove from pool
    X_pool = X_pool[indices_keep]
    y_pool = y_pool[indices_keep]

    # If we got good performance, we are done
    if any(y_candidate > 99.0 ): # :)
      #print("We found some good candidate! :)")
      #print(y_candidate)
      break
    else:
      #print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
      #print(y_candidate)
        pass

    # If not, sample points and retrain
    X_train = np.vstack([X_train, X_candidate])
    y_train = np.concatenate([y_train, y_candidate])
    model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

    # And also add the time it takes to train, which can be important
    total_time += training_time

  #print(f"The total time for the optimization was {total_time} for a total of {n_experiments} experiments!")
  return n_experiments, n_iter

# %%

bo_above_old(q=3, seed=666, max_iterations=5) 

# %%

bo_above(q=3, seed=666, max_iterations=5) 

# %% [markdown]
# Get baseline results with fixed `q`

# %%
max_batch_size = 10  # 10
n_seeds = 10         # 10
max_iterations = 100  # 100

q_arr = range(2, max_batch_size+1)

timings_all = np.zeros((len(q_arr),n_seeds,  2))
for q in q_arr:
    timings_all[q] = [bo_above(q=q, seed=seed, max_iterations=max_iterations) for seed in range(n_seeds)]
  

#save timings
np.save("timings_all_baseline.npy", timings_all)
timings_all_mean = timings_all.mean(axis=1)
timings_exps = timings_all_mean
print("1st output", timings_exps)

# %% [markdown]
# Plots
# 



if False:
 
    rt_arr = np.linspace(0.1,1.0,5) # time of retraining as % of experiment baseline time
    ot_arr = np.linspace(0.1,1.0,5) # overhead time per experiment as % of experiment baseline time

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

    for n_exp, n_iter in timings_exps :
        x, y = np.meshgrid(rt_arr, ot_arr)
        tt_arr = compute_cost(x, y, n_exp, n_iter)
        plt.xlabel("Retraining cost %")
        plt.ylabel("Overhead cost %")
        plt.pcolormesh(rt_arr, ot_arr, tt_arr)
        plt.title('Total time as a function of %overhead and %training')
        plt.colorbar()
        plt.show()


    z = np.zeros((len(q_arr),25))
    x, y = np.meshgrid(rt_arr, ot_arr)
    for i, (n_exp, n_iter) in enumerate(timings_exps) :
        p = q_arr
        z[i,:] = compute_cost(x,y,n_exp,n_iter).flatten()
        plt.plot(p,z)
        plt.title('Total time as a function of q')
        plt.legend()
        plt.show()


#plot distribution of timings for each batch size
#standart devivation when running different seeds is huge, the optimial batch size depends on the seed (or equavvalently on the
#trajectory of the BO)


def bo_above_flex_batch(q_arr, seed, max_iterations=100):

  model, X_train, y_train, X_pool, y_pool = init_stuff(seed)

  # Count experiments
  n_experiments = 0
  
  for i in range(max_iterations):
    q = q_arr[i] if i<len(q_arr) else q_arr[-1]
    is_found, n_experiments_incr, model, X_train, y_train, X_pool, y_pool = bo_inner(model, sampler, bounds_norm, q, X_train, y_train, X_pool, y_pool)
    n_experiments += n_experiments_incr
    if is_found is True:
      break

  return n_experiments, i+1

# %% [markdown]
# Try different ways to change `q` 

# %%

# q_arr = np.arange(10,1,-2)
# q_arr = np.arange(7,1,-1)



    
q_arr = [7, 7] # np.arange(10,1,-1)


timings_all = np.zeros((n_seeds, 2))
for seed in range(n_seeds):
    timings_all[seed] = bo_above_flex_batch(q_arr, seed=seed, max_iterations=max_iterations)
    #bo_above(q=i, seed=seed, max_iterations=max_iterations)
    #bo_above_flex_batch(q_arr, seed=seed, max_iterations=max_iterations)
np.save("timings_all_compare.npy", timings_all)
print(timings_all)
timings_all_mean = timings_all.mean(axis=0)
timings_exps = timings_all_mean
print("2nd output", timings_exps)
#compute average