# best_batchers
Bayesian Optimization Hackathon for Chemistry and Materials, Project 15: Adaptive Batch Sizes for Bayesian Optimization of Reaction Yield
Project overview and goal:

https://github.com/AC-BO-Hackathon/ac-bo-hackathon.github.io/blob/main/_projects/project-15-best_batchers.md

Some Guidelines to consider

1) This repo is the main repo of our project. If you have your own idea or solution it would be most effective to share it in the discord channel and create your own branch and finally merge it with this main branch. 
2) However, Feel free to self organize however you wish!
3) Use the `init_data.py` script to test your suggested approach. This way we make sure all our suggestions are comparible with each other use the same split and initialization.
See details on this script below!
4) double check the function that tracks the time needed to perform the experiments and fit, this is trivial but critical but benchmarking
5) Have fun! :)


USE THE INITIALIZATION SCRIPT:

`init_data.py`

example of use:

```
DATASET = Evaluation_data()

(
    X_init,
    y_init,
    X_candidate,
    y_candidate, 
    LIGANDS_INIT,
    LIGANDS_HOLDOUT,
    exp_init,
    exp_holdout,
) = DATASET.get_init_holdout_data()
```

We initialize with the complete set of experiments (all solvents and temperatures) for a single ligand. `X_init` and `y_init` are the representation vector of a reaction using ECFP4 and yields respectively used to initialize the BO.

`X_candidate` and `y_candidate` are the holdout set corresponding to experiments not yet performed. `LIGANDS_INIT` is the list of ligands included, `LIGANDS_HOLDOUT` are the ligands not yet included.

`exp_init` is the list of initial experimental conditions - `exp_holdout` again corresponds to the holdout set.




