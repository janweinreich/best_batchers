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

        indices_init = self.where_worst_ligand[:200]
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

### DATASET = Evaluation_data()
### bounds_norm = DATASET.bounds_norm
### 
### (
###     X_init,
###     y_init,
###     X_pool_fixed,
###     y_pool_fixed,
###     LIGANDS_INIT,
###     LIGANDS_HOLDOUT,
###     exp_init,
###     exp_holdout,
### ) = DATASET.get_init_holdout_data(777)

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

# Check splits

# print(X_init.shape, y_init.shape, X_candidate.shape, y_candidate.shape)
#####
#####model, _ = update_model(
#####    X_init, y_init, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True
#####)
#####
#####
#####y_pred = model.posterior(X_pool).mean.detach().flatten().numpy()
#####y_std = np.sqrt(model.posterior(X_pool).variance.detach().flatten().numpy())
#####
#####y_pred = y_pred.flatten()
#####y_candidate = y_pool.flatten()
#####
#####X_train = np.copy(X_init)
#####y_train = np.copy(y_init)
#####X_pool = np.copy(X_pool_fixed)
#####y_pool = np.copy(y_pool_fixed)
#####
###### Start a timer
#####fixed_time = 1
#####overhead_time = overhead_factor * fixed_time
#####training_time = 2
#####total_time = 0
#####
###### Count experiments
#####n_experiments = 0
#####
#####for _ in range(100):
#####
#####  # Add fixed time
#####  total_time += fixed_time
#####
#####  # Select batch size, for now dummy as 5
#####  # Here is where the fun begins! Including the calls below
#####  q = 5
#####
#####  # Set up aqf
#####  qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
#####  X_candidate, _ = optimize_acqf_discrete(
#####    acq_function=qNEI,
#####    bounds=bounds_norm,
#####    q=q,
#####    choices=X_pool,
#####    unique=True,
#####    num_restarts=NUM_RESTARTS,
#####    raw_samples=RAW_SAMPLES,
#####    sequential=False,
#####  )
#####
#####  # See how selected experiments look at the moment
#####  y_pred = model.posterior(X_candidate).mean.detach().flatten().numpy()
#####  y_std = np.sqrt(model.posterior(X_candidate).variance.detach().flatten().numpy())
#####
#####  # See how they actually look
#####  indices = find_indices(X_pool, X_candidate)
#####  indices_keep = np.setdiff1d(np.arange(X_pool.shape[0]), indices)
#####  y_candidate = y_pool[indices]
#####
#####  # We have sampled the candidates, so we pay the overhead time n-1 times
#####  total_time += overhead_time*(y_candidate.shape[0] -1)
#####
#####  # We also count the number of experiments conducted
#####  n_experiments += y_candidate.shape[0]
#####
#####  # Remove from pool
#####  X_pool = X_pool[indices_keep]
#####  y_pool = y_pool[indices_keep]
#####
#####  # If we got good performance, we are done
#####  if any(y_candidate > 99.0 ): # :)
#####    print("We found some good candidate! :)")
#####    print(y_candidate)
#####    break
#####  else:
#####    print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
#####    print(y_candidate)
#####
#####  # If not, sample points and retrain
#####  X_train = np.vstack([X_train, X_candidate])
#####  y_train = np.concatenate([y_train, y_candidate])
#####  model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)
#####
#####  # And also add the time it takes to train, which can be important
#####  total_time += training_time
#####
#####print(f"The total time for the optimization was {total_time} for a total of {n_experiments} experiments!")
#####
def bo_above(q, seed):

  # Initialize data from dataset
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

  for _ in range(100):

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
      print("We found some good candidate! :)")
      print(y_candidate)
      break
    else:
      print(f"The best we could do in this selected batch was {max(y_candidate)}! :(")
      print(y_candidate)

    # If not, sample points and retrain
    X_train = np.vstack([X_train, X_candidate])
    y_train = np.concatenate([y_train, y_candidate])
    model, _ = update_model(X_train, y_train, bounds_norm, kernel_type="Tanimoto", fit_y=False, FIT_METHOD=True)

    # And also add the time it takes to train, which can be important
    total_time += training_time

  print(f"The total time for the optimization was {total_time} for a total of {n_experiments} experiments!")
  return n_experiments, n_iter

q_arr = range(3,10)
#example = bo_above(3)
timings_all =  np.zeros([10,7,2])
timings_all[0] = [bo_above(i, 10) for i in q_arr]
timings_all[1] = [bo_above(i, 1) for i in q_arr]
timings_all[2] = [bo_above(i, 2) for i in q_arr]
timings_all[3] = [bo_above(i, 3) for i in q_arr]
timings_all[4] = [bo_above(i, 4) for i in q_arr]
timings_all[5] = [bo_above(i, 5) for i in q_arr]
timings_all[6] = [bo_above(i, 6) for i in q_arr]
timings_all[7] = [bo_above(i, 7) for i in q_arr]
timings_all[8] = [bo_above(i, 8) for i in q_arr]
timings_all[9] = [bo_above(i, 9) for i in q_arr]

timings_all_mean = timings_all.mean(axis=0)
timings_exps = timings_all_mean



rt_arr = np.linspace(0.1,1.0,5) # time of retraining as % of experiment baseline time
ot_arr = np.linspace(0.1,1.0,5) # overhead time per experiment as % of experiment baseline time
from matplotlib import pyplot as plt
def compute_cost(rt, ot, n_exp, n_iter):
  total_cost = 0.0
  total_cost += n_iter  # Baseline experiment cost (per iteration)
  total_cost += rt * n_iter # Retraining cost (per iteration)
  total_cost += (n_exp - n_iter) * ot # Sum of the overheads
  return total_cost

j=0
for n_exp, n_iter in timings_exps :
  x, y = np.meshgrid(rt_arr, ot_arr)
  tt_arr = compute_cost(x, y, n_exp, n_iter)
  plt.xlabel("Retraining cost %")
  plt.ylabel("Overhead cost %")
  plt.pcolormesh(rt_arr, ot_arr, tt_arr)
  plt.title('Total time as a function of %overhead and %training')
  plt.colorbar()
  #plt.show()
  plt.savefig(f"{j}_rt-ot_map.svg")
  j+=1

plt.figure()
z = np.zeros((len(q_arr),25))
x, y = np.meshgrid(rt_arr, ot_arr)
for i, (n_exp, n_iter) in enumerate(timings_exps) :
  p = q_arr
  z[i,:] = compute_cost(x,y,n_exp,n_iter).flatten()
plt.plot(p,z)
plt.title('Total time as a function of q')
plt.legend()
#plt.show()
plt.savefig('time-vs-q_plot.svg')
