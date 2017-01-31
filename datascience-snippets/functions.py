"""
The :mod:`datascience-snippets.functions module includes various functions
that have not yet been wrapped together to form an own submodule.
"""

import numpy as np
import pandas as pd


def gridScoresToDF(lnT, ddof=0):
    """Convert a list of named tuples (as provided by the grid_scores_ module
    of sklearn's GridSearchCV) to a pandas DataFrame for easier post processing.

    Parameters
    ----------

    lnT : list of _CVScoreTuple
          Object returned by sklearn.grid_search.GridSearchCV.grid_scores_

    ddof : None or Int, optional
           Delta Degrees of Freedom used for computation of standard deviation.
           (From numpy.std: )
           Means Delta Degrees of Freedom. The divisor used in calculations
           is N - ddof, where N represents the number of elements.
           By default ddof is zero.
    """
    params = list(lnT[0].parameters)
    df = pd.DataFrame(index=np.arange(len(lnT)),
                      columns=['mean', 'std', 'scores'] + params)
    for i in range(len(lnT)):
        namedTuple = lnT[i]
        df.ix[i, 'mean'] = namedTuple.mean_validation_score
        df.ix[i, 'std'] = np.std(namedTuple.cv_validation_scores, ddof=ddof)
        df.ix[i, 'scores'] = namedTuple.cv_validation_scores
        for p in params:
            df.ix[i, p] = namedTuple.parameters[p]

    return df
