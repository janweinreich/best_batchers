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