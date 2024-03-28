import numpy as np
from BO_utils import update_model
from datasets import Evaluation_data
from exp_configs_1 import benchmark
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random

np.random.seed(777)
random.seed(777)


exp_config = benchmark[0]

DATASET = Evaluation_data(
    exp_config["dataset"],
    exp_config["ntrain"],
    "random",
    init_strategy=exp_config["init_strategy"],
)
bounds_norm = DATASET.bounds_norm

(
    X_init,
    y_init,
    costs_init,
    X_candidate,
    y_candidate,
    costs_candidate,
) = DATASET.get_init_holdout_data(777)


fit_y = False

model, scaler_y = update_model(
    X_init, y_init, bounds_norm, kernel_type="Linear", fit_y=fit_y, FIT_METHOD=True
)


if fit_y:
    y_pred = scaler_y.inverse_transform(
        model.posterior(X_candidate).mean.detach()
    ).flatten()
    y_std = np.sqrt(
        scaler_y.inverse_transform(
            model.posterior(X_candidate).variance.detach()
        ).flatten()
    )

else:
    y_pred = model.posterior(X_candidate).mean.detach().flatten().numpy()
    y_std = np.sqrt(model.posterior(X_candidate).variance.detach().flatten().numpy())

y_pred = y_pred.flatten()
y_candidate = y_candidate.numpy().flatten()


r2 = r2_score(y_pred, y_candidate)
mae = mean_absolute_error(y_candidate, y_pred)


print("train size = ", len(X_init))
print("test size = ", len(X_candidate))
print("r2 = ", r2)
print("N = ", len(X_init), "MAE = ", mae)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))


if exp_config["dataset"] == "BMS":
    dataset = "C-H Acrylation"
elif exp_config["dataset"] == "buchwald":
    dataset = "Buchwald"
elif exp_config["dataset"] == "baumgartner":
    dataset = "Baumgartner"
else:
    raise ValueError("Unknown dataset")

ax.set_title("{} GPR".format(dataset))
ax.errorbar(y_candidate, y_pred, yerr=y_std, marker=None, fmt=",", alpha=0.1)
ax.plot(y_candidate, y_candidate, color="black", alpha=0.2)
ax.scatter(y_candidate, y_pred, alpha=0.6)
# Setting the axis limits
ax.set_xlim(0, 102)
ax.set_ylim(0, 102)
ax.text(
    5, 95, f"R² score: {r2:.2f}", fontsize=12
)  # Adjust position and fontsize as needed
ax.text(5, 90, f"MAE: {mae:.2f}", fontsize=12)  # Adjust position and fontsize as needed
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("Experiment", fontsize=12, fontweight="bold")
ax.set_ylabel("Prediction", fontsize=12, fontweight="bold")
plt.savefig(f"correlation_{dataset}_GPR.pdf")
