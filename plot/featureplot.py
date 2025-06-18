import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vector
from matplotlib.lines import Line2D


vector.register_awkward()

DEFAULT_LABELS = {
    "jet_pt": "Jet $p_T$ [GeV]",
    "jet_eta": "Jet $\\eta$",
    "jet_phi": "Jet $\\phi$",
}
def binclip(x, bins, dropinf=False):
    binfirst_center = bins[0] + (bins[1] - bins[0]) / 2
    binlast_center = bins[-2] + (bins[-1] - bins[-2]) / 2
    if dropinf:
        print("Dropping inf")
        print("len(x) before:", len(x))
        x = x[~np.isinf(x)]
        print("len(x) after:", len(x))
    return np.clip(x, binfirst_center, binlast_center)


def plot_features(
    ak_array_dict,
    names=None,
    label_prefix=None,
    flatten=True,
    histkwargs=None,
    legend_only_on=None,
    legend_kwargs={},
    ax_rows=1,
    decorate_ax_kwargs={},
    bins_dict=None,
    logscale_features=None,
    colors=None,
    ax_size=(3, 2),
):
    """Plot the features of the constituents or jets.

    Parameters:
    ----------
    ak_array_dict : dict of awkward array
        Dict with {"name": ak.Array, ...} of the constituents or jets to plot.
    names : list of str or dict, optional
        Names of the features to plot. Either a list of names, or a dict of {"name": "label", ...}.
    label_prefix : str, optional
        Prefix for the plot x-axis labels.
    flatten : bool, optional
        Whether to flatten the arrays before plotting. Default is True.
    histkwargs : dict, optional
        Keyword arguments passed to plt.hist.
    legend_only_on : int, optional
        Plot the legend only on the i-th subplot. Default is None.
    legend_kwargs : dict, optional
        Keyword arguments passed to ax.legend.
    ax_rows : int, optional
        Number of rows of the subplot grid. Default is 1.
    decorate_ax_kwargs : dict, optional
        Keyword arguments passed to `decorate_ax`.
    bins_dict : dict, optional
        Dict of {name: bins} for the histograms. `name` has to be the same as the keys in `names`.
    logscale_features : list, optional
        List of features to plot in log scale, of "all" to plot all features in log scale.
    colors : list, optional
        List of colors for the histograms. Has to have the same length as the number of arrays.
        If shorter, the colors will be repeated.
    ax_size : tuple, optional
        Size of the axes. Default is (3, 2).
    """

    default_hist_kwargs = {"density": True, "histtype": "step", "bins": 100}

    # setup colors
    if colors is not None:
        if len(colors) < len(ak_array_dict):
            print(
                "Warning: colors list is shorter than the number of arrays. "
                "Will use default colors for remaining ones."
            )
            colors = colors + [f"C{i}" for i in range(len(ak_array_dict) - len(colors))]

    if histkwargs is None:
        histkwargs = default_hist_kwargs
    else:
        histkwargs = default_hist_kwargs | histkwargs

    # create the bins dict
    if bins_dict is None:
        bins_dict = {}
    # loop over all names - if the name is not in the bins_dict, use the default bins
    for name in names:
        if name not in bins_dict:
            bins_dict[name] = histkwargs["bins"]

    # remove default bins from histkwargs
    histkwargs.pop("bins")

    if isinstance(names, list):
        names = {name: name for name in names}

    ax_cols = len(names) // ax_rows + 1 if len(names) % ax_rows > 0 else len(names) // ax_rows

    fig, axarr = plt.subplots(
        ax_rows, ax_cols, figsize=(ax_size[0] * ax_cols, ax_size[1] * ax_rows)
    )
    if len(names) == 1:
        axarr = [axarr]
    else:
        axarr = axarr.flatten()

    legend_handles = []
    legend_labels = []

    for i_label, (label, ak_array) in enumerate(ak_array_dict.items()):
        color = colors[i_label] if colors is not None else f"C{i_label}"
        legend_labels.append(label)
        for i, (feat, feat_label) in enumerate(names.items()):
            if flatten:
                values = ak.flatten(getattr(ak_array, feat))
            else:
                values = getattr(ak_array, feat)

            if not isinstance(bins_dict[feat], int):
                values = binclip(values, bins_dict[feat])

            _, _, patches = axarr[i].hist(values, **histkwargs, bins=bins_dict[feat], color=color)
            axarr[i].set_xlabel(
                feat_label if label_prefix is None else f"{label_prefix} {feat_label}"
            )
            if i == 0:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=patches[0].get_edgecolor(),
                        lw=patches[0].get_linewidth(),
                        label=label,
                        linestyle=patches[0].get_linestyle(),
                    )
                )

    legend_kwargs["handles"] = legend_handles
    legend_kwargs["labels"] = legend_labels
    legend_kwargs["frameon"] = False
    for i, (_ax, feat_name) in enumerate(zip(axarr, names.keys())):
        if legend_only_on is None:
            _ax.legend(**legend_kwargs)
        else:
            if i == legend_only_on:
                _ax.legend(**legend_kwargs)

        if (logscale_features is not None and feat_name in logscale_features) or (
            logscale_features == "all"
        ):
            _ax.set_yscale("log")

    fig.tight_layout()
    return fig, axarr


def plot_features_pairplot(
    arr,
    names=None,
    pairplot_kwargs={},
    input_type="ak_constituents",
):
    """Plot the features of the constituents or jets using a pairplot.

    Parameters:
    ----------
    arr : awkward array or numpy array
        Constituents or jets.
    part_names : list or dict, optional
        List of names of the features to plot, or dict of {"name": "label", ...}.
    pairplot_kwargs : dict, optional
        Keyword arguments passed to sns.pairplot.
    input_type : str, optional
        Type of the input array. Can be "ak_constituents", "ak_jets", or "np_flat".
        "ak_constituents" is an awkward array of jet constituents of shape `(n_jets, <var>, n_features)`.
        "ak_jets" is an awkward array of jets of shape `(n_jets, n_features)`.
        "np_flat" is a numpy array of shape `(n_entries, n_features)`


    Returns:
    --------
    pairplot : seaborn.axisgrid.PairGrid
        Pairplot object of the features.
    """

    if isinstance(names, list):
        names = {name: name for name in names}

    sns.set_style("dark")
    # create a dataframe from the awkward array
    if input_type == "ak_constituents":
        df = pd.DataFrame(
            {feat_label: ak.flatten(getattr(arr, feat)) for feat, feat_label in names.items()}
        )
    elif input_type == "ak_jets":
        df = pd.DataFrame({feat_label: getattr(arr, feat) for feat, feat_label in names.items()})
    elif input_type == "np_flat":
        df = pd.DataFrame(
            {feat_label: arr[:, i] for i, (feat, feat_label) in enumerate(names.items())}
        )
    else:
        raise ValueError(f"Invalid input_type: {input_type}")
    pairplot = sns.pairplot(df, kind="hist", **pairplot_kwargs)
    plt.show()

    # reset the style
    plt.rcdefaults()

    return pairplot