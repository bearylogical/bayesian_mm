from copy import deepcopy
from pathlib import Path
from typing import Union
import logging
import corner
import emcee
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import arviz as az
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# use Latex for the labels in plots
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 22})

from src.inference.evaluate import get_psrf
from src.inference.sampler import BaseSampler, BayesSampler

logger = logging.getLogger("bayesian_nn")

default_capillary_kwargs = dict(
    img_size=(600, 400),
    taper_cutoff=200,
    is_deg=False,
    fill_alpha_inner=0.8,
    fill_alpha_outer=0,
)

rng = np.random.default_rng(42)

cm = plt.get_cmap("tab20")


def plot_trace(
    sampler: BaseSampler = None,
    show_psrf: bool = False,
    show_chains: bool = False,
    save_path: Path = None,
):
    samples = sampler.get_chain()
    n_samples, n_chains, ndims = samples.shape
    labels = sampler.param_names

    _, axes = plt.subplots(ndims, figsize=(10, 7), sharex=True)

    for i in range(ndims):
        ax = axes[i]
        if show_chains:
            for c in range(n_chains):
                ax.plot(samples[:, c, i], color=cm(c), alpha=0.3, label=f"Chain {c+1}")
        else:
            ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.legend(loc="lower right", fontsize=13)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    if show_psrf:
        psrfs = []
        for i in range(2, n_samples):
            psrfs.append(get_psrf(sampler, i))
        for dim in range(ndims):
            ax = axes[dim]
            ax2 = ax.twinx()
            _psrfs = [p[dim] for p in psrfs]
            ax2.plot(range(2, n_samples), _psrfs, "r--", alpha=0.3, label="PSRF")
            ax2.set_ylabel("PSRF")
            ax2.set_ylim(0, 3)
            ax2.legend(fontsize=15)
            ax.autoscale(tight=True)

    if save_path:
        plt.savefig(save_path / "traceplot.png")

    plt.show()


def plot_auto_corr(sampler: BaseSampler, save_path: Path = None):

    labels = sampler.param_names

    def autocorr_new(y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += emcee.autocorr.function_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = emcee.autocorr.auto_window(taus, c)
        return taus[window]

    N = np.exp(np.linspace(np.log(10), np.log(len(sampler.get_chain())), 20)).astype(
        int
    )
    ndim = 2
    ests = np.empty(shape=(ndim, len(N)))

    for dim in range(ndim):
        chain = sampler.get_chain()[:, :, dim].T
        for i, n in enumerate(N):
            ests[dim, i] = autocorr_new(chain[:, :n])

    _, axes = plt.subplots(1, figsize=(10, 7), sharex=True)

    # Plot the comparisons
    for dim in range(ndim):
        axes.loglog(N, ests[dim], "o-", color=cm(dim), label=labels[dim])
    ylim = plt.gca().get_ylim()
    axes.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    axes.set_ylim(ylim)
    axes.set_xlabel("Samples")
    axes.set_ylabel(r"$\tau$ estimates")
    axes.legend(fontsize=14)

    if save_path:
        plt.savefig(save_path / "traceplot.png")


def plot_confidence_ellipse(means, cov, ax, n_std=1, **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor="red",
        facecolor="none",
        **kwargs,
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = means[0], means[1]
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_corner(
    sampler: BaseSampler = None,
    truths: list = None,
    save_path: Path = None,
    quantiles=[0.16, 0.5, 0.84],
):
    assert sampler is not None
    flat_samples = sampler.get_chain()
    bins = 20
    nsamples, _ = flat_samples.shape

    fig = corner.corner(
        flat_samples,
        bins=bins,
        labels=sampler.param_names,
        truths=truths,
        quantiles=quantiles,
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )

    # loop over the histogram if param_dists exists

    if sampler.param_dists is not None:
        for i, dist in enumerate(sampler.param_dists):
            ax = fig.axes[i * sampler.n_params + i]
            params = np.linspace(*ax.get_xlim(), 100)
            params_space = params[1] - params[0]
            space_prob = dist.cdf(params + 0.5 * params_space) - dist.cdf(
                params - 0.5 * params_space
            )
            probs = nsamples * (100 / bins) * space_prob
            ax.plot(params, probs, "b--")

    if save_path:
        plt.savefig(save_path / "cornerplot.png")

    plt.show()


def plot_posterior(
    sampler: emcee.EnsembleSampler, labels: list, save_path: Path = None
):
    idata = az.from_emcee(sampler, var_names=labels)

    az.plot_posterior(idata, kind="hist")

    if save_path:
        plt.savefig(save_path / "posterior_plot.png")

    plt.show()


def plot_psrf(sampler: BayesSampler, save_path: Path = None):
    cm = plt.get_cmap("tab10")
    labels = sampler.param_names
    _, axes = plt.subplots(1, figsize=(10, 7), sharex=True)

    n_samples, _, ndims = sampler.get_chain().shape

    psrfs = []
    for i in range(2, n_samples):
        psrfs.append(get_psrf(sampler, i))
    for dim in range(ndims):
        _psrfs = [p[dim] for p in psrfs]
        axes.plot(
            range(2, n_samples),
            _psrfs,
            ls="--",
            color=cm(dim),
            alpha=1,
            label=labels[dim],
        )
    axes.axhline(
        y=1.1, xmin=0, xmax=n_samples, ls="-", color="red", label=r"Target $R_c$"
    )
    axes.set_ylabel("PSRF")
    axes.set_xlabel("Samples")
    axes.set_ylim(1, 2)
    axes.set_xlim(0, 10000)
    axes.legend(fontsize=15)

    if save_path:
        plt.savefig(save_path / "psrf_plot.png")

    plt.show()


def plot_acf(sampler: emcee.EnsembleSampler, labels: list, save_path: Path = None):
    cm = plt.get_cmap("tab10")

    def autocorr_new(y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += emcee.autocorr.function_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = emcee.autocorr.auto_window(taus, c)
        return taus[window]

    N = np.exp(np.linspace(np.log(10), np.log(len(sampler.get_chain())), 20)).astype(
        int
    )
    ndim = 2
    ests = np.empty(shape=(ndim, len(N)))

    for dim in range(ndim):
        chain = sampler.get_chain()[:, :, dim].T
        for i, n in enumerate(N):
            ests[dim, i] = autocorr_new(chain[:, :n])
    _, axes = plt.subplots(1, figsize=(10, 7), sharex=True)

    # Plot the comparisons
    for dim in range(ndim):
        axes.loglog(N, ests[dim], "o-", color=cm(dim), label=labels[dim])
    ylim = plt.gca().get_ylim()
    axes.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    axes.set_ylim(ylim)
    axes.set_xlabel("Samples")
    axes.set_ylabel(r"$\tau$ estimates")
    axes.legend(fontsize=14)

    if save_path:
        plt.savefig(save_path / "autocorr_1.png")

    plt.show()


def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1e-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ",k", alpha=0.1)
    ax.set_xlabel(r"G")
    ax.set_ylabel(r"K")


def plot_results(
    x,
    y,
    sampler: BaseSampler,
    truths: list = None,
    show_map: bool = True,
    show_bayesian_ci: bool = True,
    show_prediction_interval: bool = True,
    show_mle: bool = True,
    show_mle_ci: bool = True,
    titles: list = [
        r"Plot of $\epsilon_v$ against $\sigma_c$",
        r"Plot of $\epsilon_s$ against $\sigma_s$",
    ],
    xlabels: list = [r"$\epsilon_v$", r"$\epsilon_s$"],
    ylabels: list = [r"$\sigma_c$ [kPa]", r"$\sigma_s$ [kPa]"],
):

    # MLE stuff here
    mle, err = sampler.max_likelihood(x, y)
    mle_output = sampler.model_fn(mle[:2], x)

    _, axes = plt.subplots(ncols=2, figsize=(20, 8))
    samples = sampler.get_samples().samples
    lower_param, upper_param = np.percentile(samples, [2.2, 97.7], axis=0)
    map_val, lower_bound, upper_bound = sampler.predict(samples, x)

    for idx in range(2):
        ax = axes.flat[idx]
        if show_mle:
            ax.plot(
                x.T[idx],
                mle_output.T[idx],
                "r-",
                label="Maximum likelihood estimate (MLE)",
            )
        if show_mle_ci:
            ax.plot(
                x.T[idx],
                x.T[idx] * (mle[idx] - err[idx] * 2),
                "k--",
                label="(MLE) 95\% Confidence Interval",
            )
            ax.plot(x.T[idx], x.T[idx] * (mle[idx] + err[idx] * 2), "k--")
        if truths is not None:
            ax.plot(
                x.T[idx],
                x.T[idx] * truths[idx],
                "-",
                color="magenta",
                label="Ground Truth",
            )
        if show_map:
            ax.plot(
                x.T[idx],
                map_val[idx] * x.T[idx],
                "-",
                color="blue",
                label="Maximum a posteriori (MAP)",
            )
        if show_bayesian_ci:
            ax.fill_between(
                x.T[idx],
                x.T[idx] * lower_param[idx],
                x.T[idx] * upper_param[idx],
                alpha=0.4,
                color="c",
                label=r"(Bayesian) 95\% Credible Interval",
            )
        if show_prediction_interval:
            ax.plot(
                x.T[idx],
                lower_bound[idx],
                "--",
                color="grey",
                label=r"(Bayesian) 95\% Prediction Interval",
            )
            ax.plot(x.T[idx], upper_bound[idx], "--", color="grey")

        # show our data
        ax.scatter(x.T[idx], y.T[idx])
        ax.set_title(titles[idx], pad=20)
        ax.set_xlabel(xlabels[idx])
        ax.set_ylabel(ylabels[idx])
        ax.legend(fontsize=12)
        # axes[0].set_xlim(0, .25)
        ax.autoscale(tight=True)
    plt.show()
    # plt.savefig('assets/overall_1.svg',  bbox_inches='tight')
