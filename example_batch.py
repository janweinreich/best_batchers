# short code snipped to show how batched qNEI works 
import numpy as np

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler
from botorch.models import gp_regression

from init_data import *

NUM_RESTARTS = 20
RAW_SAMPLES = 512
sampler = SobolQMCNormalSampler(1024)

DATASET = Evaluation_data()
print(DATASET.X)
print(DATASET.experiments.shape)
print(np.matrix(DATASET.experiments[:5]))
bounds_norm = DATASET.bounds_norm

(
    X_init,
    y_init,
    X_candidate,
    y_candidate,
    LIGANDS_INIT,
    LIGANDS_HOLDOUT,
    exp_init,
    exp_holdout,
) = DATASET.get_init_holdout_data(SEED)

exit()

model = gp_regression.SingleTaskGP()

qNEI = qNoisyExpectedImprovement(model, torch.tensor(X_train), sampler)
candidates, _ = optimize_acqf_discrete(
    acq_function=qLogNEI,
    bounds=bounds_norm,
    q=q,
    choices=X_candidate_BO,
    unique=True,
    num_restarts=NUM_RESTARTS,
    raw_samples=RAW_SAMPLES,
    sequential=False,
)
