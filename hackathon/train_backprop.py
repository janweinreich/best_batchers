# An example of using BOSS for hyperparameter optimization.
import glob
import os
import random

import gpytorch
import numpy as np
import torch

# for installation of boss run: pip install aalto-boss
random.seed(1)
np.random.seed(1)


# Replace with path to QM9 directory.
QM9_dir = os.environ["DATA"] + "/QM9_formatted"

quant_name = "HOMO eigenvalue"
seed = 1

train_num = 1000
hyperparam_opt_num = 500
test_num = 1000

size = 29  # maximum number of atoms in a QM9 molecule.


xyz_list = glob.glob(QM9_dir + "/*.xyz")
random.seed(seed)
random.shuffle(xyz_list)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kern):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kern)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    from qml2.jit_interfaces import set_default_jit

    set_default_jit("torch")
    from torch import set_default_device

    from qml2.jit_interfaces.torch_interface import recommended_device

    set_default_device(recommended_device())
    from qml2.compound import Compound
    from qml2.dataset_formats.qm9 import Quantity
    from qml2.gpytorch import construct_GlobalKernel
    from qml2.jit_interfaces import array_
    from qml2.kernels.kernels import half_inv_sq_sigma  # needs rework
    from qml2.representations import generate_coulomb_matrix

    def get_quants_comps(xyz_list, quantity):
        quant_vals = array_([quantity.extract_xyz(xyz_file) for xyz_file in xyz_list])
        comps = [Compound(xyz_file) for xyz_file in xyz_list]
        representations = array_(
            [
                generate_coulomb_matrix(comp.nuclear_charges, comp.coordinates, size)
                for comp in comps
            ]
        )
        return representations, quant_vals

    quant = Quantity(quant_name)
    # Training and test sets.
    training_reps, training_quants = get_quants_comps(xyz_list[:train_num], quant)
    test_reps, test_quants = get_quants_comps(xyz_list[-test_num:], quant)

    kern = construct_GlobalKernel(type="exp", metric="l2", sigma_to_param=half_inv_sq_sigma)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(training_reps, training_quants, likelihood, kern)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(training_reps)
        # Calc loss and backprop gradients
        loss = -mll(output, training_quants)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
            % (
                i + 1,
                i,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()


if __name__ == "__main__":
    main()
