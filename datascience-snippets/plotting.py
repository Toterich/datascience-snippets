"""
The :mod:`datascience-snippets.plotting module includes functions to quickly
plot data in various ways.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import math

import numpy as np


def scatterGridSearchResults(data, params=None, title=None, randomized=False):
    """Show optimized parameters of a gridsearch and their mean score in a grid
    of scatterplots.

    Parameters
    ----------

    data : GridScore Dataframe

    params : Iterable of Strings
             The parameters to plot. Elements must be column names of data.
             If None, plot all parameters of data.

    title : String or None, optional
            The title to be displayed above the plotgrid.

    """

    # get number of optimized parameters
    if params is None:
        params = [x for x in data.columns.tolist() if x not in ['mean', 'std', 'scores']]

    nrOfParams = len(params)

    # Replace NA with "None" so that it can be plotted
    data.fillna('None', inplace=True)

    dim = math.ceil(nrOfParams ** 0.5)
    f, ax = plt.subplots(dim, dim, sharey=True)
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,6), sharey=True)
    paramIdx = 0
    for i in range(dim):
        for j in range(dim):
            if paramIdx < nrOfParams:
                sns.stripplot(x=params[paramIdx], y='mean', data=data, ax=ax[i][j])
                if randomized:
                    dSeries = data[params[paramIdx]]
                    if len(dSeries.unique()) >= 8:
                        try:
                            ticks = np.linspace(min(data[params[paramIdx]]),
                                                max(data[params[paramIdx]]),
                                                num=7).astype(int)
                            ax[i][j].set_xticks(ticks)
                        except TypeError:  # If data is not numeric
                            pass
                paramIdx += 1
            else:
                ax[i][j].axis('off')

    f.suptitle(title, size=16)
    f.tight_layout()
    f.subplots_adjust(top=0.88)

    return f
