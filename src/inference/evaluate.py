import emcee
import scipy.stats as stats
import numpy as np


def kl_divergence(p1: np.ndarray, p2: np.ndarray):
    """

    Parameters
    ----------
    p1 : proposal distribution (prior)
    p2 : target distribution (posterior)

    Returns
    -------

    """
    return stats.entropy(p1, p2, base=2)


def get_psrf(sampler: emcee.EnsembleSampler, length_chain: int):
    chains = sampler.get_chain()[:length_chain, :, :]
    _, n_chains, _ = chains.shape

    chain_mean = chains.mean(axis=0)
    between_chain_var = np.var(chain_mean, ddof=1, axis=0)
    B = length_chain * between_chain_var
    within_chain_var = np.var(chains, ddof=1, axis=0)
    W = np.mean(within_chain_var, axis=0)
    r_hat = (length_chain - 1) / length_chain + (1 + n_chains) / (
        length_chain * n_chains
    ) * (B / W)

    return r_hat


def get_acf(x, max_lag=10):
    acf_lags = np.zeros(max_lag)
    for lag in range(max_lag):
        # Slice the relevant subseries based on the lag
        y1 = x[: (len(x) - lag)]
        y2 = x[lag:]
        # Subtract the mean of the whole series x to calculate Cov
        sum_product = np.sum((y1 - np.mean(x)) * (y2 - np.mean(x)))
        # Normalize with var of whole series
        acf_lags[lag] = sum_product / ((len(x) - lag) * np.var(x))

    return acf_lags


def coefficient_of_variance(samples: np.ndarray):
    return samples.std(axis=0) / samples.mean(axis=0)


def posterior_predictive(params, inputs, noise, model):
    _predicted_y = []
    for p, n in zip(params, noise):
        rl_model_output = model(p, inputs)
        _predicted_y.append(stats.norm.rvs(loc=rl_model_output, scale=np.sqrt(n)))
    predicted_ys = np.stack(_predicted_y)
    predicted_means = np.array(
        [np.mean(predicted_ys[:, :, i], axis=0) for i in range(2)]
    )
    predicted_sd = np.array([np.std(predicted_ys[:, :, t], axis=0) for t in range(2)])
    lower_bound = predicted_means - 2 * predicted_sd
    upper_bound = predicted_means + 2 * predicted_sd
    return lower_bound, upper_bound


def get_prediction_interval(x, y, params, model=None, ci=0.95):
    """Obtain the 95% (default) prediction intervals from a given model and observed data.

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    params : _type_
        _description_
    model : _type_, optional
        _description_, by default DefaultModel()
    ci : float, optional
        Confidence interval in percentage, between 0 and 1, by default 0.95

    Returns
    -------
    _type_
        _description_
    """
    assert x.ndim == 2 and y.ndim == 2, "dimensions of x and y needs to be 2."
    assert ci > 0 and ci < 1, "CI has to be between 0 and 1"
    assert model is not None, "No model defined"
    #  Calculate model fit:
    y_fit = model(params, x)
    #  Calculate the residuals:
    resid = y - y_fit
    #  Estimate the standard errors:
    sigma2_est = np.std(resid, ddof=2, axis=0)
    y_pred_lower = []
    y_pred_upper = []
    for n in range(x.ndim):
        # Calculate the prediction SE:
        y_pred_se = 1 / np.dot(x[:, n], x[:, n].T)
        y_pred_se = np.dot(np.dot(x[:, n], y_pred_se), x[:,n].T)
        y_pred_se = np.identity(len(x[:, n])) + y_pred_se
        y_pred_se = sigma2_est[n] * y_pred_se
        y_pred_se = np.sqrt(np.diag(y_pred_se))
        # Prediction intervals for the predicted Y:
        _y_pred_lower = (
            y_fit[:, n] - stats.t.ppf(q=1 - (1 - ci) / 2, df=len(y) - 2) * y_pred_se
        )
        _y_pred_upper = (
            y_fit[:, n] + stats.t.ppf(q=1 - (1 - ci) / 2, df=len(y) - 2) * y_pred_se
        )
        y_pred_lower.append(_y_pred_lower)
        y_pred_upper.append(_y_pred_upper)

    return y_pred_lower, y_pred_upper
