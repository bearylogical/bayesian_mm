import numpy as np

from src.inference.sampler import default_model

SEED = 42
rng = np.random.default_rng(SEED)


def generate_data_v2(
    num_obs: int = 5, num_experiments: int = 1, model=default_model, **kwargs
):
    G, K = kwargs.get("G"), kwargs.get("K")
    # define our inputs
    eps_strains_1 = np.linspace(0, 0.2, num_obs)
    eps_strains_2 = np.linspace(
        0, 0.6, num_obs
    )  # replicate what we see from real data, where G is more noticeable
    eps_strains = np.stack((eps_strains_1, eps_strains_2))

    x_experiments = np.repeat(eps_strains.T, num_experiments, axis=0)
    y_true = model([G, K], x_experiments)
    y_experiments = y_true + rng.normal(size=(num_obs * num_experiments, 2))

    return x_experiments, y_true, y_experiments
