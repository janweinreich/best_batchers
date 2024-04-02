import random
import numpy as np
random.seed(777)
np.random.seed(777)
import torch
from botorch.exceptions import InputDataWarning
import warnings
from functions import bo_varying_q
from plots import plot_results
from functions import init_stuff

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


max_batch_size = 10   # 10
n_seeds = 10          # 10
max_iterations = 100  # 100



model, X_train, y_train, X_pool, y_pool, bounds_norm = init_stuff(777)


qarr = [3, 5, 10, 15]
bo_varying_q(model, qarr, X_train, X_pool)