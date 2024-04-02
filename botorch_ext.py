import torch
from typing import Any, Tuple
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction

def optimize_acqf_discrete_modified(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    n_best: int,  # Specify how many best results to return
    max_batch_size: int = 2048,
    unique: bool = True,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    # [Existing documentation and initial checks]

    choices_batched = choices.unsqueeze(-2)

    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            with torch.no_grad():
                acq_values = _split_batch_eval_acqf(
                    acq_function=acq_function,
                    X=choices_batched,
                    max_batch_size=max_batch_size,
                )
            # Sort acq_values and get indices of the top n best values
            sorted_indices = torch.argsort(acq_values, descending=True)[:n_best]
            best_candidates = choices_batched[sorted_indices]
            best_acq_values = acq_values[sorted_indices]

            candidate_list.append(best_candidates)
            acq_value_list.append(best_acq_values)

            # Enforce uniqueness by removing the selected choices
            if unique:
                mask = torch.ones(choices_batched.shape[0], dtype=torch.bool)
                mask[sorted_indices] = False
                choices_batched = choices_batched[mask]

        # Concatenate the results
        concatenated_candidates = torch.cat(candidate_list, dim=0)
        concatenated_acq_values = torch.cat(acq_value_list, dim=0)

        # Reshape to desired format [q, n_best, -1]
        final_shape = [q, n_best, concatenated_candidates.shape[-1]]
        concatenated_candidates = concatenated_candidates.view(final_shape)

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)

        return concatenated_candidates, concatenated_acq_values

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size
        )

    sorted_indices = torch.argsort(acq_values, descending=True)[:n_best]
    best_candidates = choices_batched[sorted_indices]
    best_acq_values = acq_values[sorted_indices]

    return best_candidates, best_acq_values


def _split_batch_eval_acqf(
    acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int
) -> Tensor:
    return torch.cat([acq_function(X_) for X_ in X.split(max_batch_size)])